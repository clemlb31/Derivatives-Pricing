# Moteurs de Pricing pour Options Européennes


Cet option pricer est un projet Python complet pour le pricing d'options européennes et le calcul des Grecs. Le projet implémente deux moteurs de pricing principaux :

1. **Black-Scholes** : Formule analytique pour le calcul exact du prix théorique
2. **Monte Carlo** : Simulation stochastique pour l'estimation du prix

Le projet inclut également un calculateur complet des **Greeks** (Delta, Gamma, Vega, Theta, Rho) pour l'analyse de sensibilité et la gestion des risques.

## Fonctionnalités

### Moteurs de Pricing

- **Black-Scholes** : Pricing analytique pour options call et put européennes
- **Monte Carlo** : Simulation stochastique avec estimation de l'erreur standard
- Support des calculs vectorisés pour traiter plusieurs options simultanément
- Gestion des cas limites (options expirées, volatilité nulle, etc.)

### Calcul des Grecs

- **Delta** : Sensibilité au prix du sous-jacent
- **Gamma** : Sensibilité du Delta au prix du sous-jacent
- **Vega** : Sensibilité à la volatilité
- **Theta** : Sensibilité au temps (décroissance temporelle)
- **Rho** : Sensibilité au taux d'intérêt

### Outils supplémentaires

- Notebooks Jupyter pour l'analyse et la visualisation
- Intégration avec yfinance pour récupérer des données de marché
- Architecture modulaire et extensible basée sur Kedro

## Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation des dépendances

Clonez le repository et installez les dépendances :

```bash
# Cloner le repository (si applicable)
git clone <repository-url>
cd option-pricer

# Installer les dépendances
pip install -r requirements.txt
```

### Installation en mode développement

Pour installer le package en mode développement :

```bash
pip install -e .
```

## Utilisation

Des données réelles peuvent être téléchargé de yfinance, avec le notebook data_pulling.ipynb


**Note sur les imports** : Si le package n'est pas installé, ajoutez `src` au PYTHONPATH ou utilisez :

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pricing import BlackScholesPricer, MonteCarloPricer, GreeksCalculator
```

### Exemple basique : Pricing avec Black-Scholes

```python
from pricing import BlackScholesPricer

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
print(f"Prix du call : {call_price:.4f}")

# Calculer le prix d'un put
put_price = bs_pricer.price(S, K, T, r, sigma, option_type="put")
print(f"Prix du put : {put_price:.4f}")
```

### Exemple : Pricing avec Monte Carlo

```python
from pricing import MonteCarloPricer

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

print(f"Prix estimé : {price:.4f}")
print(f"Erreur standard : {std_error:.4f}")

# Avec intervalle de confiance
price, std_error, (lower, upper) = mc_pricer.price_with_confidence_interval(
    S, K, T, r, sigma,
    option_type="call",
    n_simulations=100000,
    confidence_level=0.95
)

print(f"Prix : {price:.4f} ± {std_error:.4f}")
print(f"Intervalle de confiance 95% : [{lower:.4f}, {upper:.4f}]")
```

### Exemple : Calcul des Grecs

```python
from pricing import GreeksCalculator

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

print("Greeks pour l'option call :")
print(f"Delta : {greeks['delta']:.4f}")
print(f"Gamma : {greeks['gamma']:.4f}")
print(f"Vega  : {greeks['vega']:.4f}")
print(f"Theta : {greeks['theta']:.4f} $/jour")
print(f"Rho   : {greeks['rho']:.4f}")

# Ou calculer individuellement
delta = greeks_calc.delta(S, K, T, r, sigma, option_type="call")
gamma = greeks_calc.gamma(S, K, T, r, sigma)
vega = greeks_calc.vega(S, K, T, r, sigma)
```

### Exemple : Black-Scholes vs Monte Carlo

```python
from pricing import BlackScholesPricer, MonteCarloPricer

# Paramètres communs
S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.05, 0.20

# Black-Scholes
bs_pricer = BlackScholesPricer()
bs_price = bs_pricer.price(S, K, T, r, sigma, option_type="call")

# Monte Carlo
mc_pricer = MonteCarloPricer(random_seed=42)
mc_price, mc_error = mc_pricer.price(S, K, T, r, sigma, option_type="call", n_simulations=100000)

print(f"Black-Scholes : {bs_price:.4f}")
print(f"Monte Carlo   : {mc_price:.4f} ± {mc_error:.4f}")
print(f"Différence    : {abs(bs_price - mc_price):.4f}")
```

### Exemple : Calculs vectorisés

```python
import numpy as np
from pricing import BlackScholesPricer

bs_pricer = BlackScholesPricer()

# Plusieurs options avec différents strikes
S = np.array([100.0, 100.0, 100.0])
K = np.array([95.0, 100.0, 105.0])  # ITM, ATM, OTM
T = np.array([0.25, 0.25, 0.25])
sigma = np.array([0.20, 0.20, 0.20])
r = 0.05

# Calculer les prix pour toutes les options
prices = bs_pricer.price_vectorized(S, K, T, r, sigma, option_type="call")

print("Prix des options :")
for i, (strike, price) in enumerate(zip(K, prices)):
    print(f"Strike {strike:.0f} : {price:.4f}")
```


## Modèles Mathématiques

### Black-Scholes

Le modèle Black-Scholes suppose que le prix de l'actif sous-jacent suit un mouvement brownien géométrique :

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

Le prix d'un call européen est donné par :

\[
C(S, K, T, r, \sigma) = S \Phi(d_1) - K e^{-rT} \Phi(d_2)
\]

où :

\[
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
\]

### Monte Carlo

La méthode Monte Carlo simule des trajectoires du prix de l'actif :

\[
S_T = S_0 \exp\left((r - \frac{\sigma^2}{2})T + \sigma\sqrt{T} Z\right)
\]

où \(Z \sim \mathcal{N}(0,1)\). Le prix est estimé comme la moyenne des payoffs actualisés :

\[
C \approx e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)
\]

### Grecs

Les Grecs sont calculés à partir des dérivées partielles du prix Black-Scholes :

- **Delta** : \(\Delta = \frac{\partial C}{\partial S} = \Phi(d_1)\)
- **Gamma** : \(\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\phi(d_1)}{S\sigma\sqrt{T}}\)
- **Vega** : \(\nu = \frac{\partial C}{\partial \sigma} = S\phi(d_1)\sqrt{T}\)
- **Theta** : \(\Theta = \frac{\partial C}{\partial t}\)
- **Rho** : \(\rho = \frac{\partial C}{\partial r} = KTe^{-rT}\Phi(d_2)\)