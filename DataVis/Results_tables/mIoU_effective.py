from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++	33.82	35.36	48.69	65.98	33.82	35.36	48.69	65.98
KPConv	73.68	93.16	96.45	98.17		92.29	96.28	98.16
Point-Transformer	89.36	83.93	79.29	88.44	86.94	73.44	79.04	71.66
Random Forest					78.00	89.93	94.42	97.56
XGBoost					76.68	85.29	94.38	95.99""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++	56.05	73.78	89.27	89.44	68.61	77.07	85.59	85.35
KPConv	80.47	83.05	87.63	91.88	83.20	85.61	86.84	92.77
Point-Transformer	74.62	89.60	83.72	91.42	62.11	55.87	62.90	76.89
Random Forest					59.21	48.77	64.57	77.32
XGBoost					48.15	43.26	58.68	79.49""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++	45.21	74.10	59.56	72.33	46.76	48.25	59.56	72.33
KPConv	63.95	79.39	87.94	95.34	67.71	81.38	87.71	92.10
Point-Transformer	30.15	61.65	66.54	83.89	76.59	77.96	75.12	66.75
Random Forest					27.64	44.68	61.08	73.50
XGBoost					27.69	42.63	60.99	75.41""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++	60.82	61.14	87.00	92.63	74.44	54.10	81.53	91.71
KPConv	52.91	68.47	90.03	90.27	57.05	73.92	86.53	93.38
Point-Transformer	91.23	81.48	75.48	88.56	68.10	47.36	76.32	89.63
Random Forest					38.29	82.15	89.95	96.59
XGBoost					32.84	70.60	90.14	95.12""",  "Montelupo")

monument = SingleExperiment("""Pointnet++	51.05	58.52	64.78	75.00	51.05	54.73	63.36	75.00
KPConv	36.05	54.22	77.22	74.99	38.19	56.29	66.63	75.00
Point-Transformer	20.68	39.31	62.47	75.20		51.24	62.47	75.00
Random Forest					39.48	54.71	71.01	75.00
XGBoost					39.29	52.97	62.49	75.00""", "Monument")

piazza = SingleExperiment("""Pointnet++	44.20	46.14	58.92	73.08	44.18	46.16	58.95	73.11
KPConv	62.00	61.47	65.72	81.69	60.54	62.54	71.12	81.76
Point-Transformer	66.54	59.05	61.89	74.98	62.01	65.53	62.70	74.86
Random Forest					48.19	44.79	64.92	76.44
XGBoost					43.60	44.19	63.34	75.06""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]