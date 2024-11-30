import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

# Télécharge les données de Bitcoin (BTC-USD) depuis Yahoo Finance pour la période de 2022 à 2024
start_date = "2022-01-01"  # Date de début
end_date = "2024-12-31"    # Date de fin
btc_data = yf.download("BTC-USD", start=start_date, end=end_date)

# Préparation des données : calcul des rendements (Returns) à partir des prix de clôture
btc_data['Returns'] = btc_data['Close'].pct_change().dropna()
btc_data = btc_data.dropna()  # Supprimer les valeurs manquantes

# Utilisation des prix de clôture et des rendements pour le HMM
X = np.column_stack([btc_data['Close'].values, btc_data['Returns'].values])

# Création et entraînement du modèle HMM
hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)
hmm_model.fit(X)

# Prédiction des états cachés
hgit remidden_states = hmm_model.predict(X)

# Ajouter les états cachés aux données pour analyse
btc_data['Hidden_State'] = hidden_states

# Visualisation des états cachés et des tendances des prix
plt.figure(figsize=(12, 8))
for i in range(hmm_model.n_components):
    state = btc_data[btc_data['Hidden_State'] == i]
    plt.plot(state.index, state['Close'], '.', label=f"State {i}")

plt.title("Bitcoin Prices with Hidden States (2022-2024)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid()
plt.show()

# Résumé des résultats : Moyennes et variances des états cachés
state_means = hmm_model.means_
state_covars = hmm_model.covars_

# Créer un DataFrame pour afficher les résultats détaillés
summary = pd.DataFrame({
    'State': range(hmm_model.n_components),
    'Mean Close Price': state_means[:, 0],
    'Mean Returns': state_means[:, 1],
    'Variance Close Price': state_covars[:, 0, 0],
    'Variance Returns': state_covars[:, 1, 1]
})

# Afficher le tableau complet dans le terminal
print("\nDetailed Analysis of HMM Model States:\n")
print(summary)

# Affichage détaillé des caractéristiques du modèle HMM dans le terminal
