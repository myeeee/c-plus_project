#pragma once

#include "AdaBoostOld.h"
#include "SegmentData.h"

using namespace std;

void CAdaBoost::Init_DAB()
{
	// メモリ確保
	m_vecWeakCl		= vector<Classifier>(LEARN_NUM * CLASS_NUM);
	m_vecClssfrWght	= vector<vector<double>>(LEARN_NUM*CLASS_NUM);
	m_vecClssfrNum	= vector<vector<int>>(CLASS_NUM);
	m_vecClResult	= vector<vector<bool>>(CLASS_NUM);
	m_vecErrRate	= vector<vector<double>>(CLASS_NUM);

	if(TREE_DEPTH == 1){
		int nDimNum = GetDimNum();
		// 特徴量の貢献度を初期化
		m_vecFtrCntrbt = vector<vector<double>>(CLASS_NUM, vector<double>(nDimNum, 0.0));
		// 出力ファイル(csv)を初期化
		fstream fsOutCsvCntrbt;
		fsOutCsvCntrbt.open(GetFoldName() + "\\FtrContribution.csv", ios::out);
		if (fsOutCsvCntrbt.is_open()){
			for(int nDim = 0; nDim < nDimNum; ++nDim)
				fsOutCsvCntrbt << nDim << ",";
			fsOutCsvCntrbt << endl;
			fsOutCsvCntrbt.close();
		}
	}

	// 識別器出力ファイル(csv)を初期化
	fstream fsOutCsvErr;
	fsOutCsvErr.open(GetFoldName() + "\\Classifier_DAB.csv", ios::out);
}

void CAdaBoost::CompSpltPnt()
{
	int nDimNum		= GetDimNum();
	int nDataNum	= GetDataNum();

	vector<vector<float>> vecSpltPnt(nDimNum, vector<float>(nDataNum-1));
	
	// コピー
	vector<vector<float>> vecSortFtr = m_vecFeature;
	// 昇順に変換
	for(int nDim = 0; nDim < nDimNum; ++nDim)
		sort(vecSortFtr[nDim].begin(), vecSortFtr[nDim].end());

	int		nCount;
	float	fSpltPnt;
	for (int nDim = 0; nDim < nDimNum; ++nDim){
		nCount = 0;
		for (int nData = 0; nData < nDataNum-1; ++nData) {
			// 各データ特徴量の中点を分割候補とする
			fSpltPnt = (vecSortFtr[nDim][nData] + vecSortFtr[nDim][nData+1]) / 2.0f;
			// 候補数削減のため四捨五入
			switch(nDim){
				// 小数第5位
				case SDCIRCLE:
				case FRNTCIRCLE:
				case TPCIRCLE:
				case DISTANCE:
					fSpltPnt = round_place(fSpltPnt, 5);
					break;
				// 小数第4位
				case LONGSIDE:		
				case NARROWSIDE:
				case HEIGHT:
				case SIDEAREA:
				case FRONTAREA:
				case TOPAREA:
				case VOLBOX:
				case MUL_LNGNRRW:
				case MUL_LNGHGHT:
				case SDDENS:
				case FRNTDENS:
				case TPDENS:
				case SDELONG:
				case FRNTELONG:
				case TPELONG:
				case MAXHEIGHT:
				case MINHEIGHT:
				case BOXSIZE_LNG:
				case BOXSIZE_NRRW:
				case MINBOXVOL:
					fSpltPnt = round_place(fSpltPnt, 4);
					break;
				// 小数第3位
				case SIDEPER:
				case FRONTPER:
				case TOPPER:
				case DIV_VARINT:
				case DIV_HGHTLNG:
				case DIV_HGHTNRRW:
				case VELOCITY:
				case VELVAR:
				case BOXAREA_SD:
				case BOXAREA_FRNT:
				case BOXAREA_TP:
					fSpltPnt = round_place(fSpltPnt, 3);
					break;
				// 小数第2位
				case INTENSITY:
				case PNTDENSITY:
				case SDDENS_NONZ:
				case FRNTDENS_NONZ:
				case TPDENS_NONZ:
				case SDELONG_NONZ:
				case FRNTELONG_NONZ:
				case TPELONG_NONZ:
				case INTMAX:
				case INTMIN:
				case SLICE_PAI:
					fSpltPnt = round_place(fSpltPnt, 2);
					break;
				// 小数点
				case POINTNUM:	
				case VARINTENSITY:
				case SD_NONZERO:
				case FRNT_NONZERO:
				case TP_NONZERO:
				case SLICE_PVI:
					fSpltPnt = round_place(fSpltPnt, 1);
					break;
				// その他の特徴量
				default:
					if(COVMTRXSTRT<=nDim && nDim<=COVMTRXEND)
						fSpltPnt = round_place(fSpltPnt, 5);
					else if(INRMTRXSTRT<=nDim && nDim<=INRMTRXEND)
						fSpltPnt = round_place(fSpltPnt, 3);
					else if(PNTHSTLNGSTRT<=nDim && nDim<=PNTHSTLNGEND)
						fSpltPnt = round_place(fSpltPnt, 4);
					else if(PNTHSTNRRWSTRT<=nDim && nDim<=PNTHSTNRRWEND)
						fSpltPnt = round_place(fSpltPnt, 4);
					else if(INTHSTSTRT<=nDim && nDim<=INTHSTEND)
						fSpltPnt = round_place(fSpltPnt, 4);
					else if(SLICE_LWSTRT<=nDim && nDim<=SLICE_LWEND)
						fSpltPnt = round_place(fSpltPnt, 4);
					else if(SLICE_NWSTRT<=nDim && nDim<=SLICE_NWEND)
						fSpltPnt = round_place(fSpltPnt, 4);
					else if(SLICE_AISTRT<=nDim && nDim<=SLICE_AIEND)
						fSpltPnt = round_place(fSpltPnt, 2);
					break;
			}
			// 新しい分割候補を保存
			if(nCount==0 || fSpltPnt!=vecSpltPnt[nDim][nCount-1]){
				vecSpltPnt[nDim][nCount] = fSpltPnt;
				++nCount;
			}
		}
		// リサイズ
		vecSpltPnt[nDim].resize(nCount);
	}
	// 保存
	m_vecSpltPnt.swap(vecSpltPnt);
}

void CAdaBoost::Training()
{
	m_strFoldName = "Training_DAB";
	_mkdir(m_strFoldName.c_str());

	// 初期化
	Init_DAB();
	// メモリ確保
	m_vecErrWght	= vector<vector<double>>(CLASS_NUM);
	m_vecSumWght	= vector<vector<double>>(CLASS_NUM);
	// データ重みの初期化
	//WeightIniOld();
	WeightIni();
	// 分割点候補の作成
	CompSpltPnt();
	// 学習
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;

		#pragma omp parallel for
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// 弱識別器の選択
			SelectClassifier(nRound, nClass);
			// 弱識別器の重みを算出
			AnalyzeClssfrWght(nRound, nClass);
			// データ重みの更新
			DataWeightUpdate(nRound, nClass);
			// 分類スコアの更新
			UpdateClScore(nRound, nClass);
			// 特徴超貢献度の計算
			if(TREE_DEPTH == 1)
				UpdateFtrCntrbt(nRound, nClass);
		}
		// 最大分類スコアによる識別結果
		UpdateClResult();
		// 閾値による識別結果
		UpdateClResult(ThrScore_DAB);
		// csvファイルにエラー率を出力
		OutCsvError(nRound);
		// csvファイルに識別器を出力
		OutCsvClssfr(nRound);
		// csvファイルに貢献度を出力
		if(TREE_DEPTH == 1)
			OutCsvFtrCntrbt(nRound);
	}
	// 識別器をdatファイルに出力
	OutDatClssfr();
	// 識別の詳細を出力
	OutCsvDetError();
	// 特徴量を出力
	OutCsvFeature();
	// Recall Precision Curve
	DrawCurve();
}

void CAdaBoost::Test()
{
	m_strFoldName = "Test_DAB";
	_mkdir(m_strFoldName.c_str());

	// 初期化
	Init_DAB();
	// 識別器の読み込み
	ReadClassifier();
	// Test
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// 決定木での分類
			CompTree(nRound, nClass);
			// 分類スコアの更新
			UpdateClScore(nRound, nClass);
			// 特徴超貢献度の計算
			if(TREE_DEPTH == 1)
				UpdateFtrCntrbt(nRound, nClass);
		}
		// 最大分類スコアによる識別結果
		UpdateClResult();
		// 閾値による識別結果
		UpdateClResult(ThrScore_DAB);
		// csvファイルにエラー率を出力
		OutCsvError(nRound);
		// csvファイルに識別器を出力
		OutCsvClssfr(nRound);
		// csvファイルに貢献度を出力
		if(TREE_DEPTH == 1)
			OutCsvFtrCntrbt(nRound);
	}
	// 識別器をdatファイルに出力
	OutDatClssfr();
	// 識別の詳細を出力
	OutCsvDetError();
	// 特徴量を出力
	OutCsvFeature();
	// Recall Precision Curve
	DrawCurve();
}

void CAdaBoost::OutCsvFtrCntrbt(int nRound)
{
	int nDimNum = GetDimNum();

	// 特徴量貢献度を出力
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\FtrContribution.csv", ios::out | ios::app);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Round" << nRound << endl;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;
			for(int nDim = 0; nDim < nDimNum; ++nDim)
				fsOutCsv << m_vecFtrCntrbt[nClass][nDim] << ",";
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}
}

void CAdaBoost::OutCsvClssfr(int nRound)
{
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Classifier_DAB.csv", ios::out | ios::app);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Round" << nRound << endl;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;
			fsOutCsv << "Decision Tree" << "\n" << "Dim, Threshold, High, Alpha" << endl;
			int i;
			for(i = 0; i < CLSSFR_NUM-CLWGHT_NUM; ++i)
				fsOutCsv << m_vecWeakCl[nRound*CLASS_NUM + nClass].TreeDim[i] << "," << m_vecWeakCl[nRound*CLASS_NUM + nClass].ThrVal[i] << "," << m_vecWeakCl[nRound*CLASS_NUM + nClass].IsHigh[i] << endl;
			for(int j = 0; j < CLWGHT_NUM; ++j)
				fsOutCsv << m_vecWeakCl[nRound*CLASS_NUM + nClass].TreeDim[i+j] << "," << m_vecWeakCl[nRound*CLASS_NUM + nClass].ThrVal[i+j] << "," << m_vecWeakCl[nRound*CLASS_NUM + nClass].IsHigh[i+j] << "," << m_vecClssfrWght[nRound*CLASS_NUM + nClass][j] << endl;
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}
}

void CAdaBoost::OutDatClssfr()
{
	fstream fsOutDatCl;
	fsOutDatCl.open(GetFoldName() + "\\Classifier_DAB.dat", ios::out | ios::binary);
	if (fsOutDatCl.is_open()){
		// 定義
		Classifier*		lpWeakCl;
		vector<int>		TreeDim;
		vector<float>	ThrVal;
		vector<int>		IsHigh;

		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				// 情報取得
				lpWeakCl = GetWeakCl(nRound, nClass);

				TreeDim = lpWeakCl->TreeDim;
				ThrVal	= lpWeakCl->ThrVal;
				IsHigh	= lpWeakCl->IsHigh;
				
				// 出力
				fsOutDatCl.write((const char*)&TreeDim[0], sizeof(int)*CLSSFR_NUM);
				fsOutDatCl.write((const char*)&ThrVal[0], sizeof(float)*CLSSFR_NUM);
				fsOutDatCl.write((const char*)&IsHigh[0], sizeof(int)*CLSSFR_NUM);
			}
		}
		fsOutDatCl.close();
	}

	//弱識別器をdatファイルに出力
	fstream fsOutDatWght;
	fsOutDatWght.open(GetFoldName() + "\\ClassifierWeight.dat", ios::out | ios::binary);
	if (fsOutDatWght.is_open()){
		for(int i = 0; i < LEARN_NUM*CLASS_NUM; ++i)
			fsOutDatWght.write((const char*)&m_vecClssfrWght[i][0], sizeof(double)*CLWGHT_NUM);
		fsOutDatWght.close();
	}
}

void CAdaBoost::ReadClassifier()
{
	//弱識別器の読み込み
	fstream ReadCl;
	ReadCl.open("Training_DAB\\Classifier_DAB.dat", ios::in | ios::binary);
	if (ReadCl.is_open()){
		vector<int>		TreeDim(CLSSFR_NUM);
		vector<float>	ThrVal(CLSSFR_NUM);
		vector<int>		IsHigh(CLSSFR_NUM);
		
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				// 読み込み
				ReadCl.read((char*)&TreeDim[0], sizeof(int)*CLSSFR_NUM);
				ReadCl.read((char*)&ThrVal[0], sizeof(float)*CLSSFR_NUM);
				ReadCl.read((char*)&IsHigh[0], sizeof(int)*CLSSFR_NUM);
				// 保存
				m_vecWeakCl[nRound*CLASS_NUM + nClass].TreeDim	= TreeDim;
				m_vecWeakCl[nRound*CLASS_NUM + nClass].ThrVal	= ThrVal;
				m_vecWeakCl[nRound*CLASS_NUM + nClass].IsHigh	= IsHigh;	
			}
		}
		ReadCl.close();
	}

	//弱識別器の重みを読み込み
	fstream fsReadWght;
	fsReadWght.open("Training_DAB\\ClassifierWeight.dat", ios::in | ios::binary);
	if (fsReadWght.is_open()){
		for(int i = 0; i < LEARN_NUM*CLASS_NUM; ++i){
			vector<double> vecClssfrWght(CLWGHT_NUM);
			// 読み込み
			fsReadWght.read((char*)&vecClssfrWght[0], sizeof(double)*CLWGHT_NUM);
			// 保存
			m_vecClssfrWght[i] = vecClssfrWght;
		}
		fsReadWght.close();
	}
}

void CAdaBoost::CompTree(int nRound, int nClass)
{
	int				nDataNum	= GetDataNum();

	vector<int>		vecClssfrNum(nDataNum);
	vector<bool>	vecClResult(nDataNum, false);

	// 決定木による分類
	vector<vector<bool>> vecTreeData;
	CompTrData(vecTreeData, nRound, nClass);	

	for(int i = 0; i < CLWGHT_NUM; ++i){
		// 弱識別器の番号
		for(int nData = 0; nData < nDataNum; ++nData){
			// Psitive
			if(vecTreeData[i*2+POS][nData]){
				// 識別結果
				vecClResult[nData] = true;
				// 弱識別器の番号
				vecClssfrNum[nData] = i;
			// Negative
			}else if(vecTreeData[i*2+NEG][nData]){
				// 弱識別器の番号
				vecClssfrNum[nData] = i;
			}
		}
	}
	// 保存
	m_vecClssfrNum[nClass].swap(vecClssfrNum);
	m_vecClResult[nClass].swap(vecClResult);
}

void CAdaBoost::CompTrData(vector<vector<bool>>& vecTreeData, int nRound, int nClass)
{
	int						nDataNum	= GetDataNum();
	Classifier*				lpWeakCl	= GetWeakCl(nRound, nClass); 
	vector<vector<float>>&	vecFeature	= GetFeature();

	// ノード数:1，全要素:trueで初期化
	vector<vector<bool>> vecOldTrData(1, vector<bool>(nDataNum, true));
	int	Count = 0;

	int		nTreeDim;
	float	fThrVal;
	int		nIsHigh;
	
	// 決定木
	for(int nDepth = 0; nDepth < TREE_DEPTH; ++nDepth){
		vector<vector<bool>> vecNewTrData(pow(2.0, nDepth+1), vector<bool>(nDataNum, false));
		for(int i = 0; i < vecOldTrData.size(); ++i){
			nTreeDim	= lpWeakCl->TreeDim[Count];
			fThrVal		= lpWeakCl->ThrVal[Count];
			nIsHigh		= lpWeakCl->IsHigh[Count];
			++Count;
			for(int nData = 0; nData < nDataNum; ++nData){
				if(!vecOldTrData[i][nData])
					continue;
				// Psitive
				if(nIsHigh == (vecFeature[nTreeDim][nData] > fThrVal))
					vecNewTrData[i*2+POS][nData] = true;
				// Negative
				else
					vecNewTrData[i*2+NEG][nData] = true;
			}
		}
		vecOldTrData.swap(vecNewTrData);
	}
	vecTreeData.swap(vecOldTrData);
}

/*
void CAdaBoost::ReadParameter()
{
	//シグモイド関数のパラメータをdatファイルに出力
	fstream ReadSigmoid;
	ReadSigmoid.open("Sigmoid Param.dat", ios::in | ios::binary);
	if (ReadSigmoid.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			ReadSigmoid.read((char*)&m_SgmdParam[nClass][0], sizeof(double)*2);
		}
		ReadSigmoid.close();
	}

	//事後確率表をdatファイルに出力
	fstream ReadProbTable;
	ReadProbTable.open("Probability Table.dat", ios::in | ios::binary);
	if (ReadProbTable.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			ReadProbTable.read((char*)&m_ScoreTbl[nClass][0], sizeof(double) * SCORE_BIN);
			ReadProbTable.read((char*)&m_ProbTbl[nClass][0], sizeof(double) * SCORE_BIN);
		}
		ReadProbTable.close();
	}

	return;
}

void CAdaBoost::ScoreToProb()
{
	ReadParameter();

	//シグモイド
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		for(int nData = 0; nData < m_DataNum; ++nData){
			m_SgmdProb[nClass][nData] = exp(m_SgmdParam[nClass][0] + m_vecH[nClass][nData]*m_SgmdParam[nClass][1]) / (1 + exp(m_SgmdParam[nClass][0] + m_vecH[nClass][nData]*m_SgmdParam[nClass][1]));
		}
	}

	//事後確率表
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		for(int nData = 0; nData < m_DataNum; ++nData){
			for(int i = 0; i < SCORE_BIN; ++i){
				if(m_vecH[nClass][nData] < m_ScoreTbl[nClass][i]){
					m_TblProb[nClass][nData] = m_ProbTbl[nClass][i];
					break;
				}
				if(i >= SCORE_BIN-1){
					m_TblProb[nClass][nData] = m_ProbTbl[nClass][i];
					break;
				}
			}
		}
	}

	return;
}

void CAdaBoost::ProbTest()
{
	InitError();

	vector<vector<double>>& vecPstrProb = m_SgmdProb;
//	vector<vector<double>>& vecPstrProb = m_TblProb;

	int tmp;
	double MaxH; //最大スコア

	for(int nData = 0; nData < m_DataNum; ++nData){
		MaxH = -DBL_MAX;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			if(vecPstrProb[nClass][nData] > MaxH){
				tmp = nClass;
				MaxH = vecPstrProb[nClass][nData];
			}else if(vecPstrProb[nClass][nData] == MaxH){
				tmp = -1;
			}
		}

		//クラス分け
		m_vecClass[nData] = tmp;
		if(tmp != -1){
			//++検出数
			++m_ClssNum[tmp];
			//++TruePositive
			if(tmp == m_vecLabel[nData]){
				++m_TrPsNum[tmp];
			}
		}
	}

	cout << "<Segment Classification (Using Probability)>" << endl;
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		cout << "--Class" << nClass << "--" << endl;
		cout << " Label:" << m_LblNum[nClass] << " Class:" << m_ClssNum[nClass] << " TruePositive:" << m_TrPsNum[nClass] << endl;
		cout << fixed << setprecision(8) << " Recall:" << (double)m_TrPsNum[nClass] / (double)m_LblNum[nClass] << " Precision:" << (double)m_TrPsNum[nClass] / (double)m_ClssNum[nClass] << endl;
	}
	
	return;
}
*/