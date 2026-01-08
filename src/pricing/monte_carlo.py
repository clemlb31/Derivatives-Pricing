"""Monte Carlo pricing engine for European options."""

import numpy as np
from typing import Literal


class MonteCarloPricer:
    """
    Moteur de pricing Monte Carlo pour les options européennes.
    
    Utilise la simulation Monte Carlo pour estimer le prix théorique
    des options call et put européennes en simulant les trajectoires
    du prix de l'actif sous-jacent selon un mouvement brownien géométrique.
    """
    
    def __init__(self, random_seed: int | None = None):
        """
        Initialise le pricer Monte Carlo.
        
        Parameters
        ----------
        random_seed : int, optional
            Graine pour la génération aléatoire (pour reproductibilité)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        n_simulations: int = 100000,
        n_steps: int = 1
    ) -> tuple[float, float]:
        """
        Calcule le prix théorique d'une option européenne avec Monte Carlo.
        
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
            Volatilité (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        n_simulations : int, optional
            Nombre de simulations Monte Carlo (par défaut: 100000)
        n_steps : int, optional
            Nombre de pas de temps pour la simulation (par défaut: 1 pour européen)
        
        Returns
        -------
        tuple[float, float]
            Tuple (prix estimé, erreur standard)
        """
        if T <= 0:
            # Option expirée
            if option_type == "call":
                payoff = max(S - K, 0)
            else:
                payoff = max(K - S, 0)
            return (payoff, 0.0)
        
        if sigma <= 0:
            # Pas de volatilité
            if option_type == "call":
                price = max(S - K * np.exp(-r * T), 0)
            else:
                price = max(K * np.exp(-r * T) - S, 0)
            return (price, 0.0)
        
        # Simulation des trajectoires du prix
        dt = T / n_steps
        
        # Génération des variables aléatoires normales
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        
        # Calcul des prix finaux selon le mouvement brownien géométrique
        # S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        if n_steps == 1:
            # Version simplifiée pour option européenne
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z[:, 0])
        else:
            # Version avec plusieurs pas de temps
            log_ST = np.log(S) + (r - 0.5 * sigma**2) * dt
            for step in range(n_steps):
                log_ST += sigma * np.sqrt(dt) * Z[:, step]
            ST = np.exp(log_ST)
        
        # Calcul du payoff
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
        
        # Actualisation et calcul de la moyenne
        discounted_payoffs = np.exp(-r * T) * payoffs
        price = np.mean(discounted_payoffs)
        
        # Calcul de l'erreur standard
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return (price, std_error)
    
    def price_with_confidence_interval(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        n_simulations: int = 100000,
        n_steps: int = 1,
        confidence_level: float = 0.95
    ) -> tuple[float, float, tuple[float, float]]:
        """
        Calcule le prix avec un intervalle de confiance.
        
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
            Volatilité (annualisée)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        n_simulations : int, optional
            Nombre de simulations Monte Carlo (par défaut: 100000)
        n_steps : int, optional
            Nombre de pas de temps pour la simulation (par défaut: 1)
        confidence_level : float, optional
            Niveau de confiance pour l'intervalle (par défaut: 0.95)
        
        Returns
        -------
        tuple[float, float, tuple[float, float]]
            Tuple (prix estimé, erreur standard, (borne inférieure, borne supérieure))
        """
        price, std_error = self.price(
            S, K, T, r, sigma, option_type, n_simulations, n_steps
        )
        
        # Calcul de l'intervalle de confiance (approximation normale)
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std_error
        
        confidence_interval = (price - margin, price + margin)
        
        return (price, std_error, confidence_interval)
    
    def price_vectorized(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: Literal["call", "put"] = "call",
        n_simulations: int = 100000,
        n_steps: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
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
            Volatilités (annualisées)
        option_type : str, optional
            Type d'option : "call" ou "put" (par défaut: "call")
        n_simulations : int, optional
            Nombre de simulations Monte Carlo (par défaut: 100000)
        n_steps : int, optional
            Nombre de pas de temps pour la simulation (par défaut: 1)
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple (prix estimés, erreurs standard)
        """
        n_options = len(S)
        prices = np.zeros(n_options)
        std_errors = np.zeros(n_options)
        
        for i in range(n_options):
            price, std_error = self.price(
                S[i], K[i], T[i], r, sigma[i], option_type, n_simulations, n_steps
            )
            prices[i] = price
            std_errors[i] = std_error
        
        return (prices, std_errors)

