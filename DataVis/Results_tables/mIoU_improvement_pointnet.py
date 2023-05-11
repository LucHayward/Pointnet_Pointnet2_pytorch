from .experiment import DoubleExperiment

bagni_nerone = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	40.88	60.84	63.68	64.38		59.92	63.45	64.36
Point-Transformer	56.97	51.12	40.8	44.92	54.48	40.08	40.46	11.36
Random Forest					45.32	57.44	60.97	63.16
XGBoost					43.96	52.56	60.91	60.02""",
                                """Pointnet++	0	0	0	0	0	0	0	0
KPConv	11.87	34.32	57.37	61.98		58.28	60.27	61.9
Point-Transformer	7.01	38.42	-16.91	44.74	31.79	31.79	31.79	31.79
Random Forest					37.83	52.97	62.44	63.25
XGBoost					48.89	53.95	63.31	59.72""",
                                "Bagni Nerone")
church = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	25.05	9.76	-2.18	4.89	14.97	8.99	1.66	14.83
Point-Transformer	19.05	16.65	-7.4	3.97	-6.66	-22.31	-30.26	-16.92
Random Forest					-9.64	-29.79	-28.03	-16.06
XGBoost					-20.98	-35.59	-35.88	-11.72""",
                          """Pointnet++	0	0	0	0	0	0	0	0
KPConv	23.01	21.5	3.89	4.28	-10.95	18.72	11.88	10.22
Point-Transformer	17.47	20.97	1.24	4.1	-49.63	-49.27	-48.61	12.16
Random Forest					-14.63	-23.68	-19.32	-15.42
XGBoost					-25.09	-28.67	-12.59	-11.73""",
                          "Church")
lunnahoja = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	19.23	5.56	37.84	46.02	21.49	34.87	37.53	39.55
Point-Transformer	-15.44	-13.11	9.31	23.12	30.6	31.27	20.74	-11.15
Random Forest					-19.61	-3.76	2.02	2.34
XGBoost					-19.55	-5.92	1.9	6.17""",
                             """Pointnet++	0	0	0	0	0	0	0	0
KPConv	5.86	-37.02	32.34	38.4	8.33	31.62	14.18	-4.35
Point-Transformer	-14.55	-12.46	-12.37	23.29	-14.55	15.14	-25.34	23.29
Random Forest					-27.94	-9.34	6.69	3.64
XGBoost					-27.96	-9.62	3.58	7.9""",
                             "Lunnahoja")
montelupo = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	-8.12	7.72	4.03	-4.73	-17.83	20.87	6.67	3.34
Point-Transformer	31.19	21.41	-15.37	-8.15	-6.5	-7.09	-6.95	-4.16
Random Forest					-37.07	29.53	11.23	9.77
XGBoost					-42.66	17.37	11.48	6.83""",
                             """Pointnet++	0	0	0	0	0	0	0	0
KPConv	-9.72	-10.02	-0.5	-23.22	-65.66	-5.09	4.52	-9.27
Point-Transformer	-5.28	12.53	-11.41	-7.95	-67.35	-31.64	-4.65	-6.15
Random Forest					-37.05	33.75	20.23	9.53
XGBoost					-46.13	24.51	15.34	6.73""",
                             "Montelupo")
monument = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	-15.38	-4.53	16.58	-0.02	-13.19	1.64	4.37	0
Point-Transformer	-31.14	-20.22	-3.08	0.4		-3.68	-1.18	0
Random Forest					-11.86	-0.02	10.21	0
XGBoost					-12.06	-1.86	-1.16	0""",
                            """Pointnet++	0	0	0	0	0	0	0	0
KPConv	-30.93	-11.17	-2.41	-0.22	-32.64	-21.47	-0.35	-11.54
Point-Transformer	-30.67	-11.39	0	-0.01		-12.24	0.09	-22.07
Random Forest					-10.48	-1.57	0.09	0
XGBoost					-13.18	-1.67	0.09	0""",
                            "Monument")
piazza = DoubleExperiment("""Pointnet++	0	0	0	0	0	0	0	0
KPConv	18.26	16.14	9.06	17.22	16.78	17.24	16.23	17.31
Point-Transformer	22.91	13.59	3.96	3.79	18.29	20.39	5.01	3.5
Random Forest					4.11	-1.45	7.96	6.67
XGBoost					-0.6	-2.08	5.86	3.91""",
                          """Pointnet++	0	0	0	0	0	0	0	0
KPConv	2.62	14.41	2.62	13.47	16.11	8.4	10.06	8.42
Point-Transformer	5.97		3.54	3.78	-27.68	-27.72	3.49	3.73
Random Forest					0.33	-4.31	2.5	5.54
XGBoost					-2.42	-4.47	1.96	3.83""",
                          "Piazza")


experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]