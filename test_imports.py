"""Script de test pour vérifier que les imports fonctionnent correctement."""

import sys
from pathlib import Path

# Ajouter le répertoire src au path pour les imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from pricing import BlackScholesPricer, MonteCarloPricer, GreeksCalculator
    print("OK Imports réussis !")
    
    # Test rapide
    bs = BlackScholesPricer()
    price = bs.price(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
    print(f"OK Test Black-Scholes : prix = {price:.4f}")
    
    mc = MonteCarloPricer(random_seed=42)
    price_mc, error = mc.price(S=100, K=105, T=0.25, r=0.05, sigma=0.20, n_simulations=10000)
    print(f"OK Test Monte Carlo : prix = {price_mc:.4f} ± {error:.4f}")
    
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.all_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
    print(f"OK Test Grecs : Delta = {greeks['delta']:.4f}")
    
    print("\nOK Tous les tests sont passés ! L'exemple devrait fonctionner.")
    
except ImportError as e:
    print(f" Erreur d'import : {e}")
    sys.exit(1)
except Exception as e:
    print(f" Erreur lors du test : {e}")
    sys.exit(1)

