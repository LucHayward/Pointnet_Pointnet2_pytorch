from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++		1.55	13.33	17.29		1.55	13.33	17.29
KPConv		19.49	3.29	1.72			3.99	1.88
Point-Transformer		-5.44	-4.63	9.15		-13.50	5.60	-7.38
Random Forest						11.93	4.49	3.14
XGBoost						8.62	9.08	1.62""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++		17.73	15.49	0.17		8.46	8.53	-0.24
KPConv		2.58	4.58	4.25		2.41	1.23	5.93
Point-Transformer		14.98	-5.88	7.70		-6.24	7.03	13.99
Random Forest						-10.44	15.80	12.75
XGBoost						-4.89	15.43	20.81""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++		28.90	-14.54	12.77		1.50	11.31	12.77
KPConv		15.43	8.56	7.40		13.67	6.33	4.39
Point-Transformer		31.50	4.89	17.34		1.37	-2.84	-8.37
Random Forest						17.05	16.39	12.42
XGBoost						14.94	18.36	14.43""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++		0.31	25.87	5.63		-20.34	27.43	10.18
KPConv		15.56	21.56	0.24		16.87	12.61	6.85
Point-Transformer		-9.76	-6.00	13.08		-20.74	28.95	13.31
Random Forest						43.86	7.80	6.64
XGBoost						37.76	19.54	4.98""",  "Montelupo")

monument = SingleExperiment("""Pointnet++		7.48	6.26	10.22		3.69	8.62	11.65
KPConv		18.17	23.00	-2.22		18.11	10.34	8.37
Point-Transformer		18.63	23.16	12.73			11.23	12.53
Random Forest						15.23	16.30	3.99
XGBoost						13.68	9.52	12.52""", "Monument")

piazza = SingleExperiment("""Pointnet++		1.93	12.79	14.16		1.98	12.78	14.16
KPConv		-0.54	4.25	15.97		2.00	8.58	10.64
Point-Transformer		-7.49	2.85	13.08		3.52	-2.83	12.15
Random Forest						-3.40	20.13	11.53
XGBoost						0.59	19.15	11.72""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]