
# Modèle de lockbox pour un signal d'erreur issu d'une spectroscopie
# d'absorption saturée (SAS — Saturated Absorption Spectroscopy).
# Le signal physique est une Lamb dip dérivée analogiquement (RC).
# Deux formes sont proposées ; lire les commentaires de SASInput pour choisir.

import numpy as np
from pyrpl.software_modules.lockbox import *

# Lorentz contient les utilitaires de forme (_lorentz, _lorentz_slope_normalized…)
# définis dans fabryperot.py. On l'importe explicitement car il n'est pas
# ré-exporté par le `from pyrpl.software_modules.lockbox import *` ci-dessus.
from pyrpl.software_modules.lockbox.models.fabryperot import Lorentz


# ==============================================================================
# Classe 1 : le signal d'entrée SAS
# ==============================================================================

class SASDeriveInput(InputSignal, Lorentz):
    """
    Physique du signal
    ------------------
    La Lamb dip est une Lorentzienne en absorption :

        L(x) = 1 / (1 + x²)   avec x = (ν - ν₀) / Γ_HWHM

    Le dérivateur RC produit une forme proportionnelle à la dérivée :

        dL/dx = -2x / (1 + x²)²    ← _lorentz_slope dans la classe Lorentz
    """

    # ------------------------------------------------------------------
    # Forme du signal attendue
    # ------------------------------------------------------------------

    def expected_signal(self, variable):
        """
        Signal attendu en Volts pour un setpoint `variable` en unité de
        la lockbox (par défaut : bandwidth = HWHM).

        Forme : A · f(x),  où f = _lorentz_slope_normalized,
        dont les extrema sont ±1 en x = ±1/√3,
        et f(0) = 0  →  le zéro correspond exactement au setpoint 0.

        calibration_data.amplitude = (max - min)/2
        est mesuré lors de la calibration sur la courbe centrée,
        de sorte que expected_signal(±1/√3) = ±amplitude.
        """
        
        x = variable * self.lockbox._setpoint_unit_in_unit("bandwidth") 
        A = self.calibration_data.amplitude
        # _lorentz_slope_normalized : extrema ±1 en x = ±1/√3, zéro en x=0
        return A * self._lorentz_slope_normalized(x) #représente le signal après soustraction de l'offset.

    def expected_slope(self, variable):
        """
        Dérivée analytique de expected_signal.
        """
        A = self.calibration_data.amplitude
        norm =3*np.sqrt(3)/8
        dxdv = self.lockbox._setpoint_unit_in_unit("bandwidth")
        return -2*A/norm*dxdv

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, autosave=False, timeout_min=1):
        """
        Calibration du signal SAS.

        Procédure
        ---------
        1. Sweep de la sortie piezo (ou autre actionneur).
        2. Acquisition d'une courbe brute sur l'entrée SAS.
        3. Estimation de la ligne de base Doppler par la médiane de la courbe.
           (La médiane est robuste si la résonance est étroite devant le sweep.)
        4. Soustraction de la baseline → courbe centrée sur zéro.
        5. Calcul de min / max / rms sur la courbe centrée.

        Après calibration
        -----------------
        calibration_data.amplitude  ≈ demi-amplitude crête-à-crête du pic SAS
        calibration_data.mean       ≈ 0  (par construction)
        calibration_data.min / max  = extrema du signal centré (≈ ±amplitude)

        La baseline est absorbée dans calibration_data._analog_offset
        (mécanisme standard PyRPL), de sorte que les lectures en temps réel
        via input.mean sont aussi centrées.
        """
        # Étape 1-2 : sweep + acquisition (méthode héritée d'InputSignal)
        curve, times = self.sweep_acquire(timeout_min=timeout_min)
        if curve is None:
            self._logger.warning("Calibration SAS annulée : oscilloscope indisponible.")
            return None

        # Étape 3 : estimation de la ligne de base Doppler
        # On utilise la médiane plutôt que la moyenne :
        # si le sweep ne couvre qu'une partie de la raie Doppler,
        # la moyenne serait biaisée par l'asymétrie.

        baseline = float(np.median(curve)) #mesure l'offset Doppler"

        # Étape 4 : centrage
        # On accumule dans _analog_offset (convention PyRPL) :
        # toutes les lectures ultérieures via input.mean seront aussi corrigées.
        self.calibration_data._analog_offset = baseline
        curve_centered = curve - baseline # On travaille ici sur la courbe centrée

        # Étape 5 : statistiques sur la courbe centrée (DONC SANS OFFSET)
        self.calibration_data.get_stats_from_curve(curve_centered)

        self._logger.info(
            "Calibration SAS réussie — "
            "baseline (offset Doppler) : %.4f V | "
            "amplitude A : %.4f V | "
            "min : %.4f V | max : %.4f V | rms : %.4f V",
            baseline,
            self.calibration_data.amplitude, #demi-amplitude de la courbe déjà centrée. L'offset a été soustrait une fois pour toutes.
            self.calibration_data.min,
            self.calibration_data.max,
            self.calibration_data.rms,
        )

        # Notifier l'interface graphique (met à jour la courbe affichée)
        self.lockbox._signal_launcher.input_calibrated.emit([self])

        if autosave:
            params = self.calibration_data.setup_attributes
            params["name"] = self.name + "_calibration"
            newcurve = self._save_curve(times, curve_centered, **params)
            self.calibration_data.curve = newcurve
            return newcurve

        return None

class SASDirectInput(InputSignal, Lorentz):
    """
    Signal SAS non dérivé : la Lamb dip (Lorentzienne en absorption).
    Utilisé pour le verrouillage grossier sur le flanc.
    Le setpoint 0 correspond au minimum de la résonance (fond du creux).
    Un setpoint de ±1/√3 ≈ ±0.577 correspond au flanc à pente maximale.
    """
    def expected_signal(self, variable):
        x = variable * self.lockbox._setpoint_unit_in_unit("bandwidth")
        A = self.calibration_data.amplitude
        # Lorentzienne en absorption : 0 au minimum, +A hors résonance
        # _lorentz(0) = 1, _lorentz(∞) = 0
        # On centre : expected_signal = 0 en dehors de la résonance,
        #             expected_signal = -A au fond du creux
        return A * (self._lorentz(x) - 1.0)

    def calibrate(self, autosave=False, timeout_min=1):
        curve, times = self.sweep_acquire(timeout_min=timeout_min)
        if curve is None:
            return None
        # Pas de soustraction de baseline ici : la ligne de base est le niveau
        # hors résonance, qui est la référence naturelle pour une Lorentzienne
        self.calibration_data.get_stats_from_curve(curve)
        self.lockbox._signal_launcher.input_calibrated.emit([self])
        return None

# ==============================================================================
# Classe 2 : la Lockbox SAS complète
# ==============================================================================

class SASLockbox(Lockbox):
    """
    Lockbox pour un asservissement sur le zéro d'un signal SAS dérivé
    analogiquement.

    Paramètres GUI
    --------------
    transition_linewidth : FWHM de la transition atomique (Hz)
    setpoint_unit        : unité du setpoint (bandwidth, Hz, MHz)

    Entrée 1  : reference  → SASDirectInput :(entrée analogique Red Pitaya)
    Entrée 2 : sas_derive -> SASDeriveInput :entrée analogique red pitaya 
    Sortie  : piezo → PiezoOutput (cale piezo laser, courant, AOM…)

    Utilisation rapide
    ------------------
    >>> p.lockbox.classname = 'SASLockbox'
    >>> p.lockbox.transition_linewidth = 6.066e6      # Rb87 D2
    >>> p.lockbox.inputs.sas.input_signal = 'in1'
    >>> p.lockbox.outputs.piezo.output_channel = 'out1'
    >>> p.lockbox.calibrate_all()
    >>> p.lockbox.lock()
    """

    variable = "detuning"

    # ------------------------------------------------------------------
    # Paramètres physiques
    # ------------------------------------------------------------------

    transition_linewidth = FrequencyProperty(
        default=6.066e6,
        min=1e3,
        max=1e12,
        doc=(
            "Largeur à mi-hauteur FWHM de la transition atomique en Hz. "
            "Rb87 D2 780 nm : 6.066 MHz | Rb85 D2 780 nm : 5.75 MHz | "
            "Cs D2 852 nm : 5.22 MHz."
        ),
    )

    # ------------------------------------------------------------------
    # Unités du setpoint
    # PyRPL cherche _unitA_in_unitB pour les conversions.
    # On définit _bandwidth_in_Hz ; les préfixes SI (MHz, kHz…) sont
    # gérés automatiquement par _unit1_in_unit2.
    # ------------------------------------------------------------------

    setpoint_unit = SelectProperty(
        options=["bandwidth", "Hz", "MHz"],
        default="bandwidth",
        doc=(
            "'bandwidth' = HWHM (demi-largeur). "
            "'Hz' ou 'MHz' = détuning absolu en fréquence. "
            "Le setpoint 0 correspond à la résonance dans toutes les unités."
        ),
    )

    _output_units = ["V", "Hz", "MHz"]

    @property
    def _bandwidth_in_Hz(self):
        """HWHM de la transition : linewidth / 2."""
        return self.transition_linewidth / 2.0

    # ------------------------------------------------------------------
    # Entrées / sorties
    # ------------------------------------------------------------------

    inputs  = LockboxModuleDictProperty(reference=SASDirectInput,sas_derive=SASDeriveInput)
    outputs = LockboxModuleDictProperty(piezo=PiezoOutput)

    # ------------------------------------------------------------------
    # Attributs sauvegardés / affichés dans le GUI
    # ------------------------------------------------------------------

    _setup_attributes = ["transition_linewidth", "setpoint_unit","inputs","outputs"]
    _gui_attributes   = ["transition_linewidth", "setpoint_unit"]
