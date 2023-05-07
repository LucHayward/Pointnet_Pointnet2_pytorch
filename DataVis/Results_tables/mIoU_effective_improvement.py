from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++	1.70	3.40	17.10	34.02	1.70	3.40	17.10	34.02
KPConv	0.67	0.36	1.18	1.83		0.41	1.24	1.84
Point-Transformer	0.27	0.85	6.90	11.56	0.33	1.40	6.99	28.34
Random Forest					0.56	0.53	1.86	2.44
XGBoost					0.60	0.77	1.88	4.01""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++	1.13	1.38	3.58	10.57	0.80	1.21	4.80	14.65
KPConv	0.50	0.89	4.12	8.12	0.43	0.76	4.39	7.24
Point-Transformer	0.65	0.55	5.43	8.58	0.97	2.32	12.37	23.11
Random Forest					1.05	2.70	11.81	22.68
XGBoost					1.33	2.99	13.77	20.51""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++	1.41	1.36	13.48	27.68	1.37	2.72	13.48	27.68
KPConv	0.92	1.08	4.02	4.67	0.83	0.98	4.10	7.90
Point-Transformer	1.79	2.02	11.15	16.12	0.60	1.16	8.30	33.25
Random Forest					1.86	2.91	12.98	26.51
XGBoost					1.85	3.02	13.01	24.59""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++	1.00	2.05	4.33	7.37	0.66	2.42	6.16	8.30
KPConv	1.21	1.66	3.33	9.74	1.10	1.37	4.49	6.63
Point-Transformer	0.22	0.97	8.18	11.45	0.82	2.77	7.90	10.38
Random Forest					1.58	0.94	3.35	3.41
XGBoost					1.72	1.55	3.29	4.88""",  "Montelupo")

monument = SingleExperiment("""Pointnet++	1.26	2.18	11.74	25.00	1.26	2.38	12.22	25.00
KPConv	1.64	2.41	7.60	25.01	1.59	2.30	11.12	25.00
Point-Transformer	2.03	3.19	12.51	24.80		2.57	12.51	25.00
Random Forest					1.55	2.38	9.66	25.00
XGBoost					1.56	2.48	12.51	25.00""", "Monument")

piazza = SingleExperiment("""Pointnet++	1.43	2.84	13.69	26.92	1.43	2.83	13.69	26.90
KPConv	0.97	2.03	11.43	18.31	1.01	1.97	9.63	18.24
Point-Transformer	0.86	2.16	12.70	25.03	0.97	1.81	12.43	25.15
Random Forest					1.33	2.91	11.70	23.56
XGBoost					1.45	2.94	12.22	24.94""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]