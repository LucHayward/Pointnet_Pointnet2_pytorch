from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++	68.68	69.64	76.31	84.02	68.68	69.64	76.31	84.02
KPConv	28.83	11.84	28.55	51.83		12.71	28.72	51.84
Point-Transformer	13.14	21.07	45.71	61.56	15.57	31.56	45.96	78.34
Random Forest					24.50	15.07	30.58	52.44
XGBoost					25.82	19.71	30.63	54.01""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++	46.45	31.22	35.73	60.57	33.90	27.93	39.41	64.65
KPConv	22.03	21.95	37.37	58.12	19.30	19.39	38.16	57.24
Point-Transformer	27.88	15.40	41.28	58.58	40.39	49.13	62.10	73.11
Random Forest					43.29	56.23	60.43	72.68
XGBoost					54.35	61.74	66.32	70.51""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++	57.30	30.90	65.44	77.68	55.74	56.75	65.44	77.68
KPConv	38.55	25.62	37.06	54.67	34.79	23.62	37.29	57.90
Point-Transformer	72.35	43.35	58.46	66.12	25.91	27.04	49.89	83.25
Random Forest					74.86	60.32	63.93	76.51
XGBoost					74.81	62.37	64.02	74.59""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++	41.68	43.86	38.00	57.37	28.06	50.90	43.47	58.30
KPConv	49.59	36.53	34.98	59.74	45.45	31.08	38.47	56.63
Point-Transformer	11.27	23.53	49.53	61.45	34.40	57.64	48.69	60.38
Random Forest					64.21	22.85	35.05	53.41
XGBoost					69.66	34.40	34.86	54.88""",  "Montelupo")

monument = SingleExperiment("""Pointnet++	51.45	46.48	60.22	75.00	51.45	50.27	61.65	75.00
KPConv	66.45	50.78	47.79	75.01	64.32	48.71	58.37	75.00
Point-Transformer	81.82	65.69	62.53	74.80		53.76	62.53	75.00
Random Forest					63.02	50.29	53.99	75.00
XGBoost					63.21	52.03	62.52	75.00""", "Monument")

piazza = SingleExperiment("""Pointnet++	58.30	58.87	66.08	76.92	58.32	58.84	66.06	76.90
KPConv	40.50	43.53	59.28	68.31	41.96	42.46	53.88	68.24
Point-Transformer	35.96	45.95	63.11	75.03	40.49	39.47	62.30	75.15
Random Forest					54.31	60.21	60.09	73.56
XGBoost					58.90	60.81	61.66	74.94""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]