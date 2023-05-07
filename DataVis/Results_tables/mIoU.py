from .experiment import DoubleExperiment

bagni_nerone = DoubleExperiment("""Pointnet++	32.12	31.96	31.59	31.96	32.12	31.96	31.59	31.96
KPConv	73	92.8	95.27	96.34		91.88	95.04	96.32
Point-Transformer	89.09	83.08	72.39	76.88	86.6	72.04	72.05	43.32
Random Forest					77.44	89.4	92.56	95.12
XGBoost					76.08	84.52	92.5	91.98""",
                                """Pointnet++	31.97	31.96	31.96	31.96	31.96	31.96	31.96	31.96
KPConv	43.84	66.28	89.33	93.94	36.54	90.24	92.23	93.86
Point-Transformer	38.98	70.38	15.05	76.7	63.75	63.75	63.75	63.75
Random Forest					69.79	84.93	94.4	95.21
XGBoost					80.85	85.91	95.27	91.68""",
                                "Bagni Nerone")

church = DoubleExperiment("""Pointnet++	54.92	72.4	85.69	78.87	67.8	75.86	80.79	70.7
KPConv	79.97	82.16	83.51	83.76	82.77	84.85	82.45	85.53
Point-Transformer	73.97	89.05	78.29	82.84	61.14	53.55	50.53	53.78
Random Forest					58.16	46.07	52.76	54.64
XGBoost					46.82	40.27	44.91	58.98""",
                          """Pointnet++	53.72	60.31	75.47	78.77	68.88	68.52	67.86	70.71
KPConv	76.73	81.81	79.36	83.05	57.93	87.24	79.74	80.93
Point-Transformer	71.19	81.28	76.71	82.87	19.25	19.25	19.25	82.87
Random Forest					54.25	44.84	48.54	55.29
XGBoost					43.79	39.85	55.27	58.98""",
                          "Church")

lunnahoja = DoubleExperiment("""Pointnet++	43.8	72.74	46.08	44.65	45.39	45.53	46.08	44.65
KPConv	63.03	78.3	83.92	90.67	66.88	80.4	83.61	84.2
Point-Transformer	28.36	59.63	55.39	67.77	75.99	76.8	66.82	33.5
Random Forest					25.78	41.77	48.1	46.99
XGBoost					25.84	39.61	47.98	50.82""",
                             """Pointnet++	44.65	72.25	44.65	44.65	44.65	44.65	44.65	44.65
KPConv	50.51	35.23	76.99	83.05	52.98	76.27	58.83	40.3
Point-Transformer	30.1	59.79	32.28	67.94	30.1	59.79	19.31	67.94
Random Forest					16.71	35.31	51.34	48.29
XGBoost					16.69	35.03	48.23	52.55""",
                             "Lunnahoja")

montelupo = DoubleExperiment("""Pointnet++	59.82	59.09	82.67	85.26	73.78	51.68	75.37	83.41
KPConv	51.7	66.81	86.7	80.53	55.95	72.55	82.04	86.75
Point-Transformer	91.01	80.5	67.3	77.11	67.28	44.59	68.42	79.25
Random Forest					36.71	81.21	86.6	93.18
XGBoost					31.12	69.05	86.85	90.24""",
                             """Pointnet++	60.93	56.56	79.99	85.22	79.63	43.92	73.23	83.42
KPConv	51.21	46.54	79.49	62	13.97	38.83	77.75	74.15
Point-Transformer	55.65	69.09	68.58	77.27	12.28	12.28	68.58	77.27
Random Forest					42.58	77.67	93.46	92.95
XGBoost					33.5	68.43	88.57	90.15""",
                             "Montelupo")

monument = DoubleExperiment("""Pointnet++	49.79	56.34	53.04	50	49.79	52.35	51.14	50
KPConv	34.41	51.81	69.62	49.98	36.6	53.99	55.51	50
Point-Transformer	18.65	36.12	49.96	50.4		48.67	49.96	50
Random Forest					37.93	52.33	61.35	50
XGBoost					37.73	50.49	49.98	50""",
                            """Pointnet++	50	48.99	50	50	50	49.84	49.91	50
KPConv	19.07	37.82	47.59	49.78	17.36	28.37	49.56	38.46
Point-Transformer	19.33	37.6	50	49.99	19.33	37.6	50	27.93
Random Forest					39.52	48.27	50	50
XGBoost					36.82	48.17	50	50""",
                            "Monument")

piazza = DoubleExperiment("""Pointnet++	42.77	43.3	45.23	46.16	42.75	43.33	45.26	46.21
KPConv	61.03	59.44	54.29	63.38	59.53	60.57	61.49	63.52
Point-Transformer	65.68	56.89	49.19	49.95	61.04	63.72	50.27	49.71
Random Forest					46.86	41.88	53.22	52.88
XGBoost					42.15	41.25	51.12	50.12""",
                          """Pointnet++	46.18	46.22	46.17	46.17	46.21	46.25	46.22	46.22
KPConv	48.8	60.63	48.79	59.64	62.32	54.65	56.28	54.64
Point-Transformer	52.15		49.71	49.95	18.53	18.53	49.71	49.95
Random Forest					46.54	41.94	48.72	51.76
XGBoost					43.79	41.78	48.18	50.05""",
                          "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]