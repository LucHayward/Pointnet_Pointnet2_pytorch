from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++		-0.95	-6.67	-7.71		-0.95	-6.67	-7.71
KPConv		16.99	-16.71	-23.28			-16.01	-23.12
Point-Transformer		-7.94	-24.63	-15.85		-16.00	-14.40	-32.38
Random Forest						9.43	-15.51	-21.86
XGBoost						6.12	-10.92	-23.39""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++		15.23	-4.51	-24.83		5.96	-11.47	-25.24
KPConv		0.08	-15.42	-20.75		-0.09	-18.77	-19.07
Point-Transformer		12.48	-25.88	-17.30		-8.74	-12.98	-11.01
Random Forest						-12.94	-4.20	-12.25
XGBoost						-7.39	-4.57	-4.19""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++		26.40	-34.54	-12.24		-1.00	-8.69	-12.24
KPConv		12.93	-11.45	-17.61		11.17	-13.67	-20.61
Point-Transformer		29.00	-15.11	-7.66		-1.13	-22.85	-33.37
Random Forest						14.55	-3.61	-12.58
XGBoost						12.44	-1.64	-10.58""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++		-2.19	5.87	-19.37		-22.84	7.43	-14.82
KPConv		13.06	1.56	-24.76		14.37	-7.39	-18.16
Point-Transformer		-12.26	-26.00	-11.92		-23.24	8.95	-11.69
Random Forest						41.36	-12.20	-18.36
XGBoost						35.26	-0.46	-20.02""",  "Montelupo")

monument = SingleExperiment("""Pointnet++		4.98	-13.74	-14.78		1.19	-11.38	-13.36
KPConv		15.67	3.00	-27.23		15.61	-9.66	-16.63
Point-Transformer		16.13	3.16	-12.27			-8.77	-12.47
Random Forest						12.73	-3.70	-21.01
XGBoost						11.18	-10.48	-12.49""", "Monument")

piazza = SingleExperiment("""Pointnet++		-0.57	-7.21	-10.84		-0.52	-7.22	-10.84
KPConv		-3.04	-15.75	-9.03		-0.50	-11.42	-14.36
Point-Transformer		-9.99	-17.15	-11.92		1.02	-22.83	-12.85
Random Forest						-5.90	0.13	-13.48
XGBoost						-1.91	-0.85	-13.28""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]