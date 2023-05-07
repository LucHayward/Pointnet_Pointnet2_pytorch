from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++	-0.15	0	0.37	0	-0.16	0	0.37	0
KPConv	-29.16	-26.52	-5.94	-2.4		-1.64	-2.81	-2.46
Point-Transformer	-50.11	-12.7	-57.34	-0.18	-22.85	-8.29	-8.3	20.43
Random Forest					-7.65	-4.47	1.84	0.09
XGBoost					4.77	1.39	2.77	-0.3""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++	-1.2	-12.09	-10.22	-0.1	1.08	-7.34	-12.93	0.01
KPConv	-3.24	-0.35	-4.15	-0.71	-24.84	2.39	-2.71	-4.6
Point-Transformer	-2.78	-7.77	-1.58	0.03	-41.89	-34.3	-31.28	29.09
Random Forest					-3.91	-1.23	-4.22	0.65
XGBoost					-3.03	-0.42	10.36	0""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++	0.85	-0.49	-1.43	0	-0.74	-0.88	-1.43	0
KPConv	-12.52	-43.07	-6.93	-7.62	-13.9	-4.13	-24.78	-43.9
Point-Transformer	1.74	0.16	-23.11	0.17	-45.89	-17.01	-47.51	34.44
Random Forest					-9.07	-6.46	3.24	1.3
XGBoost					-9.15	-4.58	0.25	1.73""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++	1.11	-2.53	-2.68	-0.04	5.85	-7.76	-2.14	0.01
KPConv	-0.49	-20.27	-7.21	-18.53	-41.98	-33.72	-4.29	-12.6
Point-Transformer	-35.36	-11.41	1.28	0.16	-55	-32.31	0.16	-1.98
Random Forest					5.87	-3.54	6.86	-0.23
XGBoost					2.38	-0.62	1.72	-0.09""",  "Montelupo")

monument = SingleExperiment("""Pointnet++	0.21	-7.35	-3.04	0	0.21	-2.51	-1.23	0
KPConv	-15.34	-13.99	-22.03	-0.2	-19.24	-25.62	-5.95	-11.54
Point-Transformer	0.68	1.48	0.04	-0.41		-11.07	0.04	-22.07
Random Forest					1.59	-4.06	-11.35	0
XGBoost					-0.91	-2.32	0.02	0""", "Monument")

piazza = SingleExperiment("""Pointnet++	3.41	2.92	0.94	0.01	3.46	2.92	0.96	0.01
KPConv	-12.23	1.19	-5.5	-3.74	2.79	-5.92	-5.21	-8.88
Point-Transformer	-13.53		0.52	0	-42.51	-45.19	-0.56	0.24
Random Forest					-0.32	0.06	-4.5	-1.12
XGBoost					1.64	0.53	-2.94	-0.07""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]