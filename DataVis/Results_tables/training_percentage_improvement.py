from .experiment import DoubleExperiment

bagni_nerone = DoubleExperiment("""Pointnet++		-0.16	-0.37	0.37		-0.16	-0.37	0.37
KPConv		19.80	2.47	1.07			3.16	1.28
Point-Transformer		-6.01	-10.69	4.49		-14.56	0.01	-28.73
Random Forest						11.93	4.49	3.14
XGBoost						8.62	9.08	1.62""",
                                """Pointnet++		-0.01	0.00	0.00		0.00	0.00	0.00
KPConv		22.44	23.05	4.61			1.99	1.63
Point-Transformer		31.40	-55.33	61.65		0.00	0.00	0.00
Random Forest						15.14	9.47	0.81
XGBoost						5.06	9.36	-3.59""",
                                "Bagni Nerone")

church = DoubleExperiment("""Pointnet++		17.48	13.29	-6.82		8.06	4.93	-10.09
KPConv		2.19	1.35	0.25		2.08	-2.40	3.08
Point-Transformer		15.08	-10.76	4.55		-7.59	-3.02	3.25
Random Forest						-12.09	6.69	1.88
XGBoost						-6.55	4.64	14.07""",
                          """Pointnet++		6.59	15.16	3.30		-0.36	-0.66	2.85
KPConv		5.08	-2.45	3.69		29.31	-7.50	1.19
Point-Transformer		10.09	-4.57	6.16		0.00	0.00	63.62
Random Forest						-9.41	3.70	6.75
XGBoost						-3.94	15.42	3.71""",
                          "Church")

lunnahoja = DoubleExperiment("""Pointnet++		28.94	-26.66	-1.43		0.14	0.55	-1.43
KPConv		15.27	5.62	6.75		13.52	3.21	0.59
Point-Transformer		31.27	-4.24	12.38		0.81	-9.98	-33.32
Random Forest						15.99	6.33	-1.11
XGBoost						13.77	8.37	2.84""",
                             """Pointnet++		27.60	-27.60	0.00		0.00	0.00	0.00
KPConv		-15.28	41.76	6.06		23.29	-17.44	-18.53
Point-Transformer		29.69	-27.51	35.66		29.69	-40.48	48.63
Random Forest						18.60	16.03	-3.05
XGBoost						18.34	13.20	4.32""",
                             "Lunnahoja")

montelupo = DoubleExperiment("""Pointnet++		-0.73	23.58	2.59		-22.10	23.69	8.04
KPConv		15.11	19.89	-6.17		16.60	9.49	4.71
Point-Transformer		-10.51	-13.20	9.81		-22.69	23.83	10.83
Random Forest						44.50	5.39	6.58
XGBoost						37.93	17.80	3.39""",
                             """Pointnet++		-4.37	23.43	5.23		-35.71	29.31	10.19
KPConv		-4.67	32.95	-17.49		24.86	38.92	-3.60
Point-Transformer		13.44	-0.51	8.69		0.00	56.30	8.69
Random Forest						35.09	15.79	-0.51
XGBoost						34.93	20.14	1.58""",
                             "Montelupo")

monument = DoubleExperiment("""Pointnet++		6.55	-3.30	-3.04		2.56	-1.21	-1.14
KPConv		17.40	17.81	-19.64		17.39	1.52	-5.51
Point-Transformer		17.47	13.84	0.44			1.29	0.04
Random Forest						14.40	9.02	-11.35
XGBoost						12.76	-0.51	0.02""",
                            """Pointnet++		-1.01	1.01	0.00		-0.16	0.07	0.09
KPConv		18.75	9.77	2.19		11.01	21.19	-11.10
Point-Transformer		18.27	12.40	-0.01			12.40	-22.07
Random Forest						8.75	1.73	0.00
XGBoost						11.35	1.83	0.00""",
                            "Monument")

piazza = DoubleExperiment("""Pointnet++		0.53	1.93	0.93		0.58	1.93	0.95
KPConv		-1.59	-5.15	9.09		1.04	0.92	2.03
Point-Transformer		-8.79	-7.70	0.76		2.68	-13.45	-0.56
Random Forest						-4.98	11.34	-0.34
XGBoost						-0.90	9.87	-1.00""",
                          """Pointnet++		0.04	-0.05	0.00		0.04	-0.03	0.00
KPConv		11.83	-11.84	10.85		-7.67	1.63	-1.64
Point-Transformer			49.71	0.24		0.00	31.18	0.24
Random Forest						-4.60	6.78	3.04
XGBoost						-2.01	6.40	1.87""",
                          "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]