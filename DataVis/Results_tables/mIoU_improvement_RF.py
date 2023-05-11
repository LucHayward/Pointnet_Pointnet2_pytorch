from experiment import DoubleExperiment

bagni_nerone = DoubleExperiment("""Pointnet++	-45.32	-57.44	-60.97	-63.16	-45.32	-57.44	-60.97	-63.16
KPConv	-4.44	3.4	2.71	1.22		2.48	2.48	1.2
Point-Transformer	11.65	-6.32	-20.17	-18.24	9.16	-17.36	-20.51	-51.8
Random Forest					0	0	0	0
XGBoost					-1.36	-4.88	-0.06	-3.14""",
                                """Pointnet++	-37.82	-52.97	-62.44	-63.25	-37.83	-52.97	-62.44	-63.25
KPConv	-25.95	-18.65	-5.07	-1.27		5.31	-2.17	-1.35
Point-Transformer	-30.81	-14.55	-79.35	-18.51	-6.04	-21.18	-30.65	-31.46
Random Forest					0	0	0	0
XGBoost					11.06	0.98	0.87	-3.53""",
                                "bagni_nerone")
church = DoubleExperiment("""Pointnet++	-3.24	26.33	32.93	24.23	9.64	29.79	28.03	16.06
KPConv	21.81	36.09	30.75	29.12	24.61	38.78	29.69	30.89
Point-Transformer	15.81	42.98	25.53	28.2	2.98	7.48	-2.23	-0.86
Random Forest					0	0	0	0
XGBoost					-11.34	-5.8	-7.85	4.34""",
                          """Pointnet++	-0.53	15.47	26.93	23.48	14.63	23.68	19.32	15.42
KPConv	22.48	36.97	30.82	27.76	3.68	42.4	31.2	25.64
Point-Transformer	16.94	36.44	28.17	27.58	-35	-25.59	-29.29	27.58
Random Forest					0	0	0	0
XGBoost					-10.46	-4.99	6.73	3.69""",
                          "Church")
lunnahoja = DoubleExperiment("""Pointnet++	18.02	30.97	-2.02	-2.34	19.61	3.76	-2.02	-2.34
KPConv	37.25	36.53	35.82	43.68	41.1	38.63	35.51	37.21
Point-Transformer	2.58	17.86	7.29	20.78	50.21	35.03	18.72	-13.49
Random Forest					0	0	0	0
XGBoost					0.06	-2.16	-0.12	3.83""",
                             """Pointnet++	27.94	36.94	-6.69	-3.64	27.94	9.34	-6.69	-3.64
KPConv	33.8	-0.08	25.65	34.76	36.27	40.96	7.49	-7.99
Point-Transformer	13.39	24.48	-19.06	19.65	13.39	24.48	-32.03	19.65
Random Forest					0	0	0	0
XGBoost					-0.02	-0.28	-3.11	4.26""",
                             "Lunnahoja")
montelupo = DoubleExperiment("""Pointnet++	23.11	-22.12	-3.93	-7.92	37.07	-29.53	-11.23	-9.77
KPConv	14.99	-14.4	0.1	-12.65	19.24	-8.66	-4.56	-6.43
Point-Transformer	54.3	-0.71	-19.3	-16.07	30.57	-36.62	-18.18	-13.93
Random Forest					0	0	0	0
XGBoost					-5.59	-12.16	0.25	-2.94""",
                             """Pointnet++	18.35	-21.11	-13.47	-7.73	37.05	-33.75	-20.23	-9.53
KPConv	8.63	-31.13	-13.97	-30.95	-28.61	-38.84	-15.71	-18.8
Point-Transformer	13.07	-8.58	-24.88	-15.68	-30.3	-65.39	-24.88	-15.68
Random Forest					0	0	0	0
XGBoost					-9.08	-9.24	-4.89	-2.8""",
                             "Montelupo")
monument = DoubleExperiment("""Pointnet++	11.86	4.01	-8.31	0	11.86	0.02	-10.21	0
KPConv	-3.52	-0.52	8.27	-0.02	-1.33	1.66	-5.84	0
Point-Transformer	-19.28	-16.21	-11.39	0.4		-3.66	-11.39	0
Random Forest					0	0	0	0
XGBoost					-0.2	-1.84	-11.37	0""",
                            """Pointnet++	10.48	0.72	0	0	10.48	1.57	-0.09	0
KPConv	-20.45	-10.45	-2.41	-0.22	-22.16	-19.9	-0.44	-11.54
Point-Transformer	-20.19	-10.67	0	-0.01		-10.67	0	-22.07
Random Forest					0	0	0	0
XGBoost					-2.7	-0.1	0	0""",
                            "Monument")
piazza = DoubleExperiment("""Pointnet++	-4.09	1.42	-7.99	-6.72	-4.11	1.45	-7.96	-6.67
KPConv	14.17	17.56	1.07	10.5	12.67	18.69	8.27	10.64
Point-Transformer	18.82	15.01	-4.03	-2.93	14.18	21.84	-2.95	-3.17
Random Forest					0	0	0	0
XGBoost					-4.71	-0.63	-2.1	-2.76""",
                          """Pointnet++	-0.36	4.28	-2.55	-5.59	-0.33	4.31	-2.5	-5.54
KPConv	2.26	18.69	0.07	7.88	15.78	12.71	7.56	2.88
Point-Transformer	5.61		0.99	-1.81	-28.01	-23.41	0.99	-1.81
Random Forest					0	0	0	0
XGBoost					-2.75	-0.16	-0.54	-1.71""",
                          "Piazza")


experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]