"""Greeks calculation for European options."""

import numpy as np
from scipy.stats import norm
from typing import Literal


class GreeksCalculator:
    """
    Calculateur des Grecs pour les options européennes.
    
    Les Grecs mesurent la sensibilité du prix d'une option aux variations
    de différents paramètres du marché. Cette classe calcule les principaux
    Grecs pour les options européennes en utilisant les formules analytiques
    dérivées du modèle Black-Scholes.
    """
    
    def __init__(self):
        """Initialise le calculateur de Grecs."""
        pass
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        """Calcule d1 et d2 pour les formules des Grecs."""
        if T <= 0 or sigma <= 0:
            return (0.0, 0.0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return (d1, d2)
    
    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call"
    ) -> float:
        """
        Calcule le Delta (sensibilité au prix du sous-jacent).
        
        Le Delta mesure la variation du prix de l'option pour une variation
        unitaire du prix de l'actif sous-jacent.
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        
        Returns
        -------
        float
            Delta de l'option
        """
        if T <= 0:
            # Option expirée
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        if sigma <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == "call":
            return norm.cdf(d1)
        else:  # put
            return -norm.cdf(-d1)
    
    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calcule le Gamma (sensibilité du Delta au prix du sous-jacent).
        
        Le Gamma mesure la variation du Delta pour une variation unitaire
        du prix de l'actif sous-jacent. Identique pour call et put.
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        
        Returns
        -------
        float
            Gamma de l'option
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calcule le Vega (sensibilité à la volatilité).
        
        Le Vega mesure la variation du prix de l'option pour une variation
        de 1% (0.01) de la volatilité. Identique pour call et put.
        Note: Le Vega est généralement divisé par 100 pour obtenir la sensibilité
        à une variation de 1% de volatilité.
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        
        Returns
        -------
        float
            Vega de l'option (pour une variation de 1.0 de volatilité)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call"
    ) -> float:
        """
        Calcule le Theta (sensibilité au temps).
        
        Le Theta mesure la variation du prix de l'option pour une diminution
        d'un jour (1/365 année) du temps jusqu'à l'expiration.
        Note: Le Theta est généralement négatif car les options perdent de la
        valeur avec le temps (décroissance temporelle).
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        
        Returns
        -------
        float
            Theta de l'option (pour une diminution de 1 jour)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365.0  # Conversion en par jour
    
    def rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call"
    ) -> float:
        """
        Calcule le Rho (sensibilité au taux d'intérêt).
        
        Le Rho mesure la variation du prix de l'option pour une variation
        de 1% (0.01) du taux d'intérêt sans risque.
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        
        Returns
        -------
        float
            Rho de l'option (pour une variation de 1.0 du taux)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call"
    ) -> dict[str, float]:
        """
        Calcule tous les Grecs principaux en une seule fois.
        
        Parameters
        ----------
        S : float
            Prix spot de l'actif sous-jacent
        K : float
            Prix d'exercice (strike)
        T : float
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : float
            Volatilité implicite (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        
        Returns
        -------
        dict[str, float]
            Dictionnaire contenant tous les Grecs :
            - delta : sensibilité au prix du sous-jacent
            - gamma : sensibilité du delta au prix du sous-jacent
            - vega : sensibilité à la volatilité (pour 1.0 de volatilité)
            - theta : sensibilité au temps (par jour)
            - rho : sensibilité au taux d'intérêt (pour 1.0 du taux)
        """
        return {
            "delta": self.delta(S, K, T, r, sigma, option_type),
            "gamma": self.gamma(S, K, T, r, sigma),
            "vega": self.vega(S, K, T, r, sigma),
            "theta": self.theta(S, K, T, r, sigma, option_type),
            "rho": self.rho(S, K, T, r, sigma, option_type)
        }

