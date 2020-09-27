import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

top50 = pd.read_csv(r'/top50')

popularity = pd.DataFrame(top50['Genre_enc'])
nam_train, nam_test, pop_train, pop_test = train_test_split(top50['Genre', 'Beats.Per.Minute', 'Energy', 'Length.',
                                                                  'Speechiness.', 'Popularity'], top50['Genre_enc'],
                                                            test_size=0.2,
                                                            random_state=0, stratify=popularity)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(nam_train, pop_train)  # Shikha lav f:X→Y
Hypothesis_prediction_y = knn.predict(nam_test)  # Antaz h:X→Y
