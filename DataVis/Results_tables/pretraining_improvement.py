from .experiment import SingleExperiment

bagni_nerone = SingleExperiment("""Pointnet++	0	0	0	0	0.01	0	0	0
KPConv		0.92	0.23	0.02		-23.96	-2.9	0.08
Point-Transformer	2.49	11.04	0.34	33.56	-24.77	6.63	-48.7	12.95""",  "Bagni Nerone")

church = SingleExperiment("""Pointnet++	-12.88	-3.46	4.9	8.17	-15.16	-8.21	7.61	8.06
KPConv	-2.8	-2.69	1.06	-1.77	18.8	-5.43	-0.38	2.12
Point-Transformer	12.83	35.5	27.76	29.06	51.94	62.03	57.46	0""",  "Church")

lunnahoja = SingleExperiment("""Pointnet++	-1.59	27.21	0	0	0	27.6	0	0
KPConv	-3.85	-2.1	0.31	6.47	-2.47	-41.04	18.16	42.75
Point-Transformer	-47.63	-17.17	-11.43	34.27	0	0	12.97	0""",  "Lunnahoja")

montelupo = SingleExperiment("""Pointnet++	-13.96	7.41	7.3	1.85	-18.7	12.64	6.76	1.8
KPConv	-4.25	-5.74	4.66	-6.22	37.24	7.71	1.74	-12.15
Point-Transformer	23.73	35.91	-1.12	-2.14	43.37	56.81	0	0""",  "Montelupo")

monument = SingleExperiment("""Pointnet++	0	3.99	1.9	0	0	-0.85	0.09	0
KPConv	-2.19	-2.18	14.11	-0.02	1.71	9.45	-1.97	11.32
Point-Transformer		-12.55	0	0.4		0	0	22.06""", "Monument")

piazza = SingleExperiment("""Pointnet++	0.02	-0.03	-0.03	-0.05	-0.03	-0.03	-0.05	-0.05
KPConv	1.5	-1.13	-7.2	-0.14	-13.52	5.98	-7.49	5
Point-Transformer	4.64	-6.83	-1.08	0.24	33.62		0	0""", "Piazza")

experiments = [bagni_nerone,
church,
lunnahoja,
montelupo,
monument,
piazza]