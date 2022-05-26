#!/bin/bash

./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 102 'CM_ENKO.Abou.010.r00.1_101.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 013 'CM_ENKO.Abou.010.r00.3_008.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 073 'CM_ENKO.Abou.013.r00.1_075.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 008 'CM_ENKO.Abou.022.r00.3_013.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 073 'CM_ENKO.Abou.023.r00.2_075.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 073 'CM_ENKO.Abou.023.r00.5_075.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 096 'CM_ENKO.Abou.031.r00.2_095.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 088 'CM_ENKO.Abou.060.r00.2_087.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 037 'CM_ENKO.Apes.001.r00.7_064.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 008 'CM_CYPR.Mvas.002.r00.4_013.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 070 'CM_ENKO.Atab.002B.r10003d.10_072.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 052 'CM_ENKO.Atab.002B.r10004d.10_049.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 086 'CMADD_RASH.Aeti.002.r00.3_064.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 112 'CMADD_RASH.Aeti.002.r00.3_064.png'

./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 025 'CMADD_IDAL.Avas.003.r00.5_027.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 061 'CMADD_KITI.Avas.021.r00.2_023.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 102 'CMADD_KITI.Avas.021.r00.1_006.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 053 'CMADD_SANI.Avas.001.r00.2_082.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 104 'CM_RASH.Atab.002.r02.3_102.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 097 'CM_MARO.Avas.001.r00.7_068.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 097 'CM_IDAL.Avas.001.r00.3_068.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 097 'CM_CYPR(QM).Psce.002.r00.1_068.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 097 'CM_ENKO.Abou.043.r00.2_068.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 097 'CM_ENKO.Abou.049.r00.2_068.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 087 'CM_KALA.Arou.001.r15.4_070.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 004 'CM_ENKO.Atab.003A.r05d.1_005.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 009 'CM_ENKO.Avas.001.r00.2_006.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 025 'CMADD_ENKO.Mins.004.r00.2_027.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 008 'CM_ENKO.Avas.004.r00.4_013.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 073 'CM_RASH.Atab.004B.r12.03_072.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 102 'CM_RASH.Atab.004A.r08.1_103.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 024 'CM_RASH.Atab.004A.r08.1_103.png'
./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 013 'CM_RASH.Atab.004A.r10.2_008.png'


#the following tests are only performed if both alternative relabeling result in a successful test and they require renaming the files in data/cyprominoan/dataset

### RASH Aeti 002 r00.3 112 vs 086 ###
#./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 112 'CMADD_RASH.Aeti.002.r00.3_086.png'

### RASH Atab 004A r08.1 102 vs 124 ###
#./single_sign_test.sh ./models/sign2vec ./data/contexts/context.csv 024 'CM_RASH.Atab.004A.r08.1_102.png'
