"""Black-Scholes pricing engine for European options."""

import numpy as np
from scipy.stats import norm
from typing import Literal


class BlackScholesPricer:
    """
    Moteur de pricing Black-Scholes pour les options européennes.
    
    Implémente la formule analytique de Black-Scholes pour le calcul
    du prix théorique des options call et put européennes.
    """
    
    def __init__(self):
        """Initialise le pricer Black-Scholes."""
        pass
    
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call"
    ) -> float:
        """
        Calcule le prix théorique d'une option européenne avec Black-Scholes.
        
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
            Prix théorique de l'option
        """
        if T <= 0:
            # Option expirée
            if option_type == "call":
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            # Pas de volatilité
            if option_type == "call":
                return max(S - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - S, 0)
        
        # Calcul des paramètres d1 et d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calcul du prix selon le type d'option
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)  # Le prix ne peut pas être négatif
    
    def d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        """
        Calcule les paramètres d1 et d2 utilisés dans la formule Black-Scholes.
        
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
        tuple[float, float]
            Tuple (d1, d2)
        """
        if T <= 0 or sigma <= 0:
            return (0.0, 0.0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return (d1, d2)
    
    def price_vectorized(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: Literal["call", "put"] = "call"
    ) -> np.ndarray:
        """
        Version vectorisée du pricing pour traiter plusieurs options simultanément.
        
        Parameters
        ----------
        S : np.ndarray
            Prix spots de l'actif sous-jacent
        K : np.ndarray
            Prix d'exercice (strikes)
        T : np.ndarray
            Temps jusqu'à l'expiration (en années)
        r : float
            Taux d'intérêt sans risque (annualisé)
        sigma : np.ndarray
            Volatilités implicites (annualisées)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        
        Returns
        -------
        np.ndarray
            Prix théoriques des options
        """
        # Gérer les cas où T <= 0 ou sigma <= 0
        valid_mask = (T > 0) & (sigma > 0)
        
        prices = np.zeros_like(S)
        
        # Cas expirés
        expired_mask = T <= 0
        if option_type == "call":
            prices[expired_mask] = np.maximum(S[expired_mask] - K[expired_mask], 0)
        else:
            prices[expired_mask] = np.maximum(K[expired_mask] - S[expired_mask], 0)
        
        # Cas sans volatilité
        no_vol_mask = (T > 0) & (sigma <= 0)
        if option_type == "call":
            prices[no_vol_mask] = np.maximum(
                S[no_vol_mask] - K[no_vol_mask] * np.exp(-r * T[no_vol_mask]), 0
            )
        else:
            prices[no_vol_mask] = np.maximum(
                K[no_vol_mask] * np.exp(-r * T[no_vol_mask]) - S[no_vol_mask], 0
            )
        
        # Cas normaux avec Black-Scholes
        normal_mask = valid_mask
        if np.any(normal_mask):
            d1 = (np.log(S[normal_mask] / K[normal_mask]) + 
                  (r + 0.5 * sigma[normal_mask]**2) * T[normal_mask]) / \
                 (sigma[normal_mask] * np.sqrt(T[normal_mask]))
            d2 = d1 - sigma[normal_mask] * np.sqrt(T[normal_mask])
            
            if option_type == "call":
                prices[normal_mask] = (
                    S[normal_mask] * norm.cdf(d1) - 
                    K[normal_mask] * np.exp(-r * T[normal_mask]) * norm.cdf(d2)
                )
            else:  # put
                prices[normal_mask] = (
                    K[normal_mask] * np.exp(-r * T[normal_mask]) * norm.cdf(-d2) - 
                    S[normal_mask] * norm.cdf(-d1)
                )
        
        return np.maximum(prices, 0)

