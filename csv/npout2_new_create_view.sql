CREATE VIEW BDO2.MB_PCUST_NPOUT_VW(MASK_YM,SUBSCTN_ID,c1,IS_NPO,R_Gender,R_AGE,MASK_AGE,R_STAR,MASK_OFFICE,IS_CHANNEL,c134,MASK_TENURE,PAY_RANK,NPIN,IS_FAKE,NP_TOT_CNT,NPIN_CHT_CNT,NPIN_TWM_CNT,NPIN_FET_CNT,NPOUT_CHT_CNT,NPOUT_TWM_CNT,NPOUT_FET_CNT,NP_IN_CARRIER_CD,c1111,c1112,c1113,OFFER_SERVICE_TYPE_CD135,OFFER_AMT,LAST_PROMO_TYPE,LAST_PROMO_NAME,LAST_REST_MM,HIS_BLRA_CNT,HIS_MOB_CNT,HANDSET_RETN_CONTR_IND,HANDSET_CONTR_EXP_MONTH_CNT,HIST_HANDSET_CONTR_CNT,HIST_HANDSET_PURCHASE_CNT,c493,c494,c495,c496,c497,c498,c499,c500,c501,IND_DESC1450,IND_DESC1451,IND_DESC124,c247,c150,c151,c152,c153,c154,c155,c156,c157,c158,c159,c191,c192,c193,c194,c195,c196,c197,c198,c199,c200,c211,c212,c213,c214,c215,c216,c217,c218,c219,c220,c221,c222,c223,c224,c225,c226,c227,c228,c229,c230,c231,c232,c233,c234,LIFE_CITY_DESC1332,LIFE_TOWN_DESC1333,LIFE_CITY_DESC1334,LIFE_TOWN_DESC1335,IND_DESC1336,IND_DESC1337,c1362,c1363,c1364,c1365,c1366,c1367,c1368,c1369,c1370,c1371,c1372,c1373,c1374,c1378,c1379,c1380,c1381,c1382,c1383,c1384,c1385,c1386,c1387,c1388,c1389,c1338,c1339,c1340,c1341,c1342,c1343,c1344,c1345,c1346,c1347,c1348,c1349,c1350,c1351,c1352,c1353,c1354,c1355,c1356,c1357,c1358,c1359,c1360,c1361,c1203,c1206,c1209,c1212,c1215,c1218,c1224,c1244,c1202,c1205,c1208,c1211,c1214,c1217,c1223,c170,c171,c172,c173,c174,c175,c176,c177,c178,c179,c201,c202,c203,c204,c205,c206,c207,c208,c209,c210,c1238,c1239,c1240,c1241,c1242,c235,c236,c237,c238,c243,c244,c245,c160,c161,c162,c163,c164,c165,c166,c167,c168,c169,c239,c240,c241,c242,MOST_MO_CARRIER,c1247,c1065,c1067,c249,c251,c1197,c1199) AS SEL  MASK_YM (TITLE 'C001') ,SUBSCTN_ID (TITLE 'C002') ,c1 (TITLE 'C003') ,IS_NPO (TITLE 'C004') ,R_Gender (TITLE 'C005') ,R_AGE (TITLE 'C006') ,MASK_AGE (TITLE 'C007') ,R_STAR (TITLE 'C008') ,MASK_OFFICE (TITLE 'C009') ,IS_CHANNEL (TITLE 'C010') ,c134 (TITLE 'C011') ,MASK_TENURE (TITLE 'C012') ,PAY_RANK (TITLE 'C013') ,NPIN (TITLE 'C014') ,IS_FAKE (TITLE 'C015') ,NP_TOT_CNT (TITLE 'C016') ,NPIN_CHT_CNT (TITLE 'C017') ,NPIN_TWM_CNT (TITLE 'C018') ,NPIN_FET_CNT (TITLE 'C019') ,NPOUT_CHT_CNT (TITLE 'C020') ,NPOUT_TWM_CNT (TITLE 'C021') ,NPOUT_FET_CNT (TITLE 'C022') ,NP_IN_CARRIER_CD (TITLE 'C023') ,c1111 (TITLE 'C024') ,c1112 (TITLE 'C025') ,c1113 (TITLE 'C026') ,OFFER_SERVICE_TYPE_CD135 (TITLE 'C027') ,OFFER_AMT (TITLE 'C028') ,LAST_PROMO_TYPE (TITLE 'C029') ,LAST_PROMO_NAME (TITLE 'C030') ,LAST_REST_MM (TITLE 'C031') ,HIS_BLRA_CNT (TITLE 'C032') ,HIS_MOB_CNT (TITLE 'C033') ,HANDSET_RETN_CONTR_IND (TITLE 'C034') ,HANDSET_CONTR_EXP_MONTH_CNT (TITLE 'C035') ,HIST_HANDSET_CONTR_CNT (TITLE 'C036') ,HIST_HANDSET_PURCHASE_CNT (TITLE 'C037') ,c493 (TITLE 'C038') ,c494 (TITLE 'C039') ,c495 (TITLE 'C040') ,c496 (TITLE 'C041') ,c497 (TITLE 'C042') ,c498 (TITLE 'C043') ,c499 (TITLE 'C044') ,c500 (TITLE 'C045') ,c501 (TITLE 'C046') ,IND_DESC1450 (TITLE 'C047') ,IND_DESC1451 (TITLE 'C048') ,IND_DESC124 (TITLE 'C049') ,c247 (TITLE 'C050') ,c150 (TITLE 'C051') ,c151 (TITLE 'C052') ,c152 (TITLE 'C053') ,c153 (TITLE 'C054') ,c154 (TITLE 'C055') ,c155 (TITLE 'C056') ,c156 (TITLE 'C057') ,c157 (TITLE 'C058') ,c158 (TITLE 'C059') ,c159 (TITLE 'C060') ,c191 (TITLE 'C061') ,c192 (TITLE 'C062') ,c193 (TITLE 'C063') ,c194 (TITLE 'C064') ,c195 (TITLE 'C065') ,c196 (TITLE 'C066') ,c197 (TITLE 'C067') ,c198 (TITLE 'C068') ,c199 (TITLE 'C069') ,c200 (TITLE 'C070') ,c211 (TITLE 'C071') ,c212 (TITLE 'C072') ,c213 (TITLE 'C073') ,c214 (TITLE 'C074') ,c215 (TITLE 'C075') ,c216 (TITLE 'C076') ,c217 (TITLE 'C077') ,c218 (TITLE 'C078') ,c219 (TITLE 'C079') ,c220 (TITLE 'C080') ,c221 (TITLE 'C081') ,c222 (TITLE 'C082') ,c223 (TITLE 'C083') ,c224 (TITLE 'C084') ,c225 (TITLE 'C085') ,c226 (TITLE 'C086') ,c227 (TITLE 'C087') ,c228 (TITLE 'C088') ,c229 (TITLE 'C089') ,c230 (TITLE 'C090') ,c231 (TITLE 'C091') ,c232 (TITLE 'C092') ,c233 (TITLE 'C093') ,c234 (TITLE 'C094') ,LIFE_CITY_DESC1332 (TITLE 'C095') ,LIFE_TOWN_DESC1333 (TITLE 'C096') ,LIFE_CITY_DESC1334 (TITLE 'C097') ,LIFE_TOWN_DESC1335 (TITLE 'C098') ,IND_DESC1336 (TITLE 'C099') ,IND_DESC1337 (TITLE 'C100') ,c1362 (TITLE 'C101') ,c1363 (TITLE 'C102') ,c1364 (TITLE 'C103') ,c1365 (TITLE 'C104') ,c1366 (TITLE 'C105') ,c1367 (TITLE 'C106') ,c1368 (TITLE 'C107') ,c1369 (TITLE 'C108') ,c1370 (TITLE 'C109') ,c1371 (TITLE 'C110') ,c1372 (TITLE 'C111') ,c1373 (TITLE 'C112') ,c1374 (TITLE 'C113') ,c1378 (TITLE 'C114') ,c1379 (TITLE 'C115') ,c1380 (TITLE 'C116') ,c1381 (TITLE 'C117') ,c1382 (TITLE 'C118') ,c1383 (TITLE 'C119') ,c1384 (TITLE 'C120') ,c1385 (TITLE 'C121') ,c1386 (TITLE 'C122') ,c1387 (TITLE 'C123') ,c1388 (TITLE 'C124') ,c1389 (TITLE 'C125') ,c1338 (TITLE 'C126') ,c1339 (TITLE 'C127') ,c1340 (TITLE 'C128') ,c1341 (TITLE 'C129') ,c1342 (TITLE 'C130') ,c1343 (TITLE 'C131') ,c1344 (TITLE 'C132') ,c1345 (TITLE 'C133') ,c1346 (TITLE 'C134') ,c1347 (TITLE 'C135') ,c1348 (TITLE 'C136') ,c1349 (TITLE 'C137') ,c1350 (TITLE 'C138') ,c1351 (TITLE 'C139') ,c1352 (TITLE 'C140') ,c1353 (TITLE 'C141') ,c1354 (TITLE 'C142') ,c1355 (TITLE 'C143') ,c1356 (TITLE 'C144') ,c1357 (TITLE 'C145') ,c1358 (TITLE 'C146') ,c1359 (TITLE 'C147') ,c1360 (TITLE 'C148') ,c1361 (TITLE 'C149') ,c1203 (TITLE 'C150') ,c1206 (TITLE 'C151') ,c1209 (TITLE 'C152') ,c1212 (TITLE 'C153') ,c1215 (TITLE 'C154') ,c1218 (TITLE 'C155') ,c1224 (TITLE 'C156') ,c1244 (TITLE 'C157') ,c1202 (TITLE 'C158') ,c1205 (TITLE 'C159') ,c1208 (TITLE 'C160') ,c1211 (TITLE 'C161') ,c1214 (TITLE 'C162') ,c1217 (TITLE 'C163') ,c1223 (TITLE 'C164') ,c170 (TITLE 'C165') ,c171 (TITLE 'C166') ,c172 (TITLE 'C167') ,c173 (TITLE 'C168') ,c174 (TITLE 'C169') ,c175 (TITLE 'C170') ,c176 (TITLE 'C171') ,c177 (TITLE 'C172') ,c178 (TITLE 'C173') ,c179 (TITLE 'C174') ,c201 (TITLE 'C175') ,c202 (TITLE 'C176') ,c203 (TITLE 'C177') ,c204 (TITLE 'C178') ,c205 (TITLE 'C179') ,c206 (TITLE 'C180') ,c207 (TITLE 'C181') ,c208 (TITLE 'C182') ,c209 (TITLE 'C183') ,c210 (TITLE 'C184') ,c1238 (TITLE 'C185') ,c1239 (TITLE 'C186') ,c1240 (TITLE 'C187') ,c1241 (TITLE 'C188') ,c1242 (TITLE 'C189') ,c235 (TITLE 'C190') ,c236 (TITLE 'C191') ,c237 (TITLE 'C192') ,c238 (TITLE 'C193') ,c243 (TITLE 'C194') ,c244 (TITLE 'C195') ,c245 (TITLE 'C196') ,c160 (TITLE 'C197') ,c161 (TITLE 'C198') ,c162 (TITLE 'C199') ,c163 (TITLE 'C200') ,c164 (TITLE 'C201') ,c165 (TITLE 'C202') ,c166 (TITLE 'C203') ,c167 (TITLE 'C204') ,c168 (TITLE 'C205') ,c169 (TITLE 'C206') ,c239 (TITLE 'C207') ,c240 (TITLE 'C208') ,c241 (TITLE 'C209') ,c242 (TITLE 'C210') ,MOST_MO_CARRIER (TITLE 'C211') ,c1247 (TITLE 'C212') ,c1065 (TITLE 'C213') ,c1067 (TITLE 'C214') ,c249 (TITLE 'C215') ,c251 (TITLE 'C216') ,c1197 (TITLE 'C217') ,c1199 (TITLE 'C218') FROM BDO2.MB_PCUST_NPOUT where MASK_YM IN ('05','06','07') and C493 IS NOT NULL AND OFFER_SERVICE_TYPE_CD135 IS NOT NULL AND OFFER_SERVICE_TYPE_CD135 <>'PPAID';