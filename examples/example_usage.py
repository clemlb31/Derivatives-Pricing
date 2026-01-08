"""
Exemple d'utilisation des moteurs de pricing d'options.

Ce script démontre comment utiliser les différents composants du projet
pour calculer les prix d'options et les Grecs.
"""

import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire src au path pour les imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pricing import BlackScholesPricer, MonteCarloPricer, GreeksCalculator


def example_black_scholes():
    """Exemple d'utilisation du moteur Black-Scholes."""
    print("=" * 60)
    print("Exemple 1 : Pricing avec Black-Scholes")
    print("=" * 60)
    
    # Initialiser le pricer
    bs_pricer = BlackScholesPricer()
    
    # Paramètres de l'option
    S = 100.0      # Prix spot
    K = 105.0      # Strike
    T = 0.25       # 3 mois (en années)
    r = 0.05       # Taux sans risque 5%
    sigma = 0.20   # Volatilité 20%
    
    # Calculer le prix d'un call
    call_price = bs_pricer.price(S, K, T, r, sigma, option_type="call")
    print(f"\nParamètres de l'option :")
    print(f"  Prix spot (S)    : {S:.2f}")
    print(f"  Strike (K)       : {K:.2f}")
    print(f"  Maturité (T)     : {T:.2f} années ({T*365:.0f} jours)")
    print(f"  Taux sans risque : {r*100:.1f}%")
    print(f"  Volatilité       : {sigma*100:.1f}%")
    
    print(f"\nPrix du call européen : {call_price:.4f}")
    
    # Calculer le prix d'un put
    put_price = bs_pricer.price(S, K, T, r, sigma, option_type="put")
    print(f"Prix du put européen  : {put_price:.4f}")
    
    # Vérifier la parité call-put
    call_put_parity = call_price - put_price - S + K * np.exp(-r * T)
    print(f"\nVérification parité call-put : {call_put_parity:.6f} (devrait être ~0)")


def example_monte_carlo():
    """Exemple d'utilisation du moteur Monte Carlo."""
    print("\n" + "=" * 60)
    print("Exemple 2 : Pricing avec Monte Carlo")
    print("=" * 60)
    
    # Initialiser le pricer Monte Carlo
    mc_pricer = MonteCarloPricer(random_seed=42)
    
    # Paramètres de l'option
    S = 100.0
    K = 105.0
    T = 0.25
    r = 0.05
    sigma = 0.20
    
    # Calculer le prix avec 100,000 simulations
    price, std_error = mc_pricer.price(
        S, K, T, r, sigma,
        option_type="call",
        n_simulations=100000
    )
    
    print(f"\nPrix estimé (Monte Carlo) : {price:.4f}")
    print(f"Erreur standard            : {std_error:.4f}")
    print(f"Intervalle de confiance 95% : [{price - 1.96*std_error:.4f}, {price + 1.96*std_error:.4f}]")
    
    # Avec intervalle de confiance
    price, std_error, (lower, upper) = mc_pricer.price_with_confidence_interval(
        S, K, T, r, sigma,
        option_type="call",
        n_simulations=100000,
        confidence_level=0.95
    )
    
    print(f"\nAvec méthode dédiée :")
    print(f"  Prix : {price:.4f} ± {std_error:.4f}")
    print(f"  Intervalle de confiance 95% : [{lower:.4f}, {upper:.4f}]")


def example_greeks():
    """Exemple de calcul des Grecs."""
    print("\n" + "=" * 60)
    print("Exemple 3 : Calcul des Grecs")
    print("=" * 60)
    
    # Initialiser le calculateur de Grecs
    greeks_calc = GreeksCalculator()
    
    # Paramètres de l'option
    S = 100.0
    K = 105.0
    T = 0.25
    r = 0.05
    sigma = 0.20
    
    # Calculer tous les Grecs
    greeks = greeks_calc.all_greeks(S, K, T, r, sigma, option_type="call")
    
    print(f"\nGreeks pour l'option call :")
    print(f"  Delta : {greeks['delta']:.4f} (sensibilité au prix du sous-jacent)")
    print(f"  Gamma : {greeks['gamma']:.4f} (sensibilité du delta)")
    print(f"  Vega  : {greeks['vega']:.4f} (sensibilité à la volatilité, pour 1.0)")
    print(f"  Theta : {greeks['theta']:.4f} $/jour (décroissance temporelle)")
    print(f"  Rho   : {greeks['rho']:.4f} (sensibilité au taux, pour 1.0)")
    
    # Calculer les Grecs pour un put
    greeks_put = greeks_calc.all_greeks(S, K, T, r, sigma, option_type="put")
    
    print(f"\nGreeks pour l'option put :")
    print(f"  Delta : {greeks_put['delta']:.4f}")
    print(f"  Gamma : {greeks_put['gamma']:.4f} (identique au call)")
    print(f"  Vega  : {greeks_put['vega']:.4f} (identique au call)")
    print(f"  Theta : {greeks_put['theta']:.4f} $/jour")
    print(f"  Rho   : {greeks_put['rho']:.4f}")


def example_comparison():
    """Comparaison entre Black-Scholes et Monte Carlo."""
    print("\n" + "=" * 60)
    print("Exemple 4 : Comparaison Black-Scholes vs Monte Carlo")
    print("=" * 60)
    
    # Paramètres communs
    S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.05, 0.20
    
    # Black-Scholes
    bs_pricer = BlackScholesPricer()
    bs_price = bs_pricer.price(S, K, T, r, sigma, option_type="call")
    
    # Monte Carlo avec différentes quantités de simulations
    mc_pricer = MonteCarloPricer(random_seed=42)
    
    print(f"\nComparaison des prix (Call, S={S}, K={K}, T={T}, r={r*100}%, σ={sigma*100}%) :")
    print(f"\nBlack-Scholes (analytique) : {bs_price:.6f}")
    print(f"\nMonte Carlo :")
    
    for n_sim in [1000, 10000, 100000, 1000000]:
        mc_price, mc_error = mc_pricer.price(
            S, K, T, r, sigma,
            option_type="call",
            n_simulations=n_sim
        )
        diff = abs(bs_price - mc_price)
        print(f"  {n_sim:>8} simulations : {mc_price:.6f} ± {mc_error:.6f} (diff: {diff:.6f})")


def example_vectorized():
    """Exemple de calculs vectorisés."""
    print("\n" + "=" * 60)
    print("Exemple 5 : Calculs vectorisés")
    print("=" * 60)
    
    bs_pricer = BlackScholesPricer()
    
    # Plusieurs options avec différents strikes
    S = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])  # ITM, ITM, ATM, OTM, OTM
    T = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
    sigma = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    r = 0.05
    
    # Calculer les prix pour toutes les options
    prices = bs_pricer.price_vectorized(S, K, T, r, sigma, option_type="call")
    
    print(f"\nPrix des options call (S={S[0]}, T={T[0]}, r={r*100}%, σ={sigma[0]*100}%) :")
    print(f"\n{'Strike':<10} {'Prix':<12} {'Moneyness':<12}")
    print("-" * 35)
    
    for strike, price in zip(K, prices):
        if strike < S[0]:
            moneyness = "ITM"
        elif strike == S[0]:
            moneyness = "ATM"
        else:
            moneyness = "OTM"
        print(f"{strike:<10.0f} {price:<12.4f} {moneyness:<12}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXEMPLES D'UTILISATION - OPTION PRICER")
    print("=" * 60)
    
    # Exécuter tous les exemples
    example_black_scholes()
    example_monte_carlo()
    example_greeks()
    example_comparison()
    example_vectorized()
    
    print("\n" + "=" * 60)
    print("FIN DES EXEMPLES")
    print("=" * 60)

