## TODO

On doit faire des codes pythons qui ajoutent chacun un indicateur au dataframe chargé depuis le parquet.
Principe : on charge le dataframe, on ajoute chaque indicateurs au dataframe, en appelant sequentiellement chaque fonction. Puis on sauvegarde le dataframe dans un nouveau parquet avec le suffixe _enhanced.parquet.

1) La moyenne mobile simple (Aymeric) -> ./Finance/MMS.py

2) La moyenne mobile exponentielle (Tom) -> ./Finance/MME.py

3) Les tendances (Aymeric) -> ./Finance/Tendance.py

4) Les volumes (Tom) -> ./Finance/Volumes.py

5) L’écart-type (Aymeric) -> ./Finance/ECT.py

6) Le MACD (Tom) -> ./Finance/MACD.py

7) Les bandes de Bollinger (Aymeric) -> ./Finance/Bollinger.py

8) Le RSI (Tom) -> ./Finance/RSI.py

9) Les retracements de Fibonacci (Aymeric) -> ./Finance/Fibonacci.py

10) Le nuage d’Ichimoku (Tom) -> ./Finance/Ichimoku.py

Noms des colonnes (avec exemple de valeur) du dataframe :
    Open Time            | 1751388060000 (2025-07-01 16:41:00)
    Open                 | 106148.26
    High                 | 106153.0
    Low                  | 106125.0
    Close                | 106152.99
    Volume               | 7.03307
    Close Time           | 1751388119999999 (Timestamp invalide/hors limites)
    Quote Volume         | 746442.9233577
    Trades               | 1377
    Taker Buy Base       | 5.14497
    Taker Buy Quote      | 546053.6489062
    Ignore               | 0