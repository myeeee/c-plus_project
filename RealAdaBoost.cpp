#pragma once

#include "RealAdaBoost.h"

using namespace std;

void CRealAdaBoost::CompSpltPnt()
{
	int nDimNum		= GetDimNum();
	int nDataNum	= GetDataNum();
	
	vector<vector<float>> vecSpltPnt(nDimNum, vector<float>(BIN[0]+BIN[1]));

	// コピー
	vector<vector<float>> vecSortFtr = m_vecFeature;
	// 昇順に変換
	for(int nDim = 0; nDim < nDimNum; ++nDim)
		sort(vecSortFtr[nDim].begin(), vecSortFtr[nDim].end());

	// (BINの幅) = (一定データ数)
	for(int i = 0; i < HIST_DIM; ++i){
		//一定データ数
		int nSpltDtNum = int(nDataNum / BIN[i]);

		for(int nDim = 0; nDim < nDimNum; ++nDim){	
			for(int nBin = 0; nBin < BIN[i]; ++nBin)
				vecSpltPnt[nDim][i*BIN[0] + nBin] = vecSortFtr[nDim][(nBin+1)*nSpltDtNum];
		}
	}
	// 保存
	m_vecSpltPnt.swap(vecSpltPnt);

	// BINをcsvファイルに出力
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Bin_RAB.csv", ios::out);
	if (fsOutCsv.is_open()){
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			fsOutCsv << "Dim" << nDim << endl;
			for(int i = 0; i < HIST_DIM; ++i){
				for(int j = 0; j < BIN[i]; ++j)
					fsOutCsv << m_vecSpltPnt[nDim][i*BIN[0] + j] << ",";
				fsOutCsv << endl;
			}
		}
		fsOutCsv.close();
	}
}

void CRealAdaBoost::Init_RAB()
{
	// メモリ確保
	m_vecWeakCl		= vector<Classifier>(LEARN_NUM * CLASS_NUM); 

	// 識別器出力ファイル(csv)を初期化
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Classifier_RAB.csv", ios::out);
}

void CRealAdaBoost::CreateFtrStrngth()
{
	int nDataNum = GetDataNum();
	int nDimNum	 = GetDimNum();
	vector<vector<float>>& vecSpltPnt = GetSpltPnt();
	vector<vector<float>>& vecFeature = GetFeature();

	vector<vector<int>> vecBinNum(HIST_DIM*nDimNum, vector<int>(nDataNum));

	size_t stBin;
	vector<float>::iterator itrStrt;
	vector<float>::iterator itrBound;
	for(int i = 0; i < HIST_DIM; ++i){
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			// BIN幅一覧のイテレータ
			itrStrt = vecSpltPnt[nDim].begin() + i*BIN[0];
			// 特徴強度を求める
			for(int nData = 0; nData < nDataNum; ++nData){
				// 二分探索
				itrBound = upper_bound(itrStrt, itrStrt+BIN[i], vecFeature[nDim][nData]);
				// BINの番号
				stBin = distance(itrStrt, itrBound);
				if(stBin > BIN[i] - 1)
					stBin = BIN[i] - 1;

				vecBinNum[i*nDimNum + nDim][nData] = stBin;
			}
		}
	}
	// 保存
	m_vecBinNum.swap(vecBinNum);
}

void CRealAdaBoost::Training()
{
	m_strFoldName = "Training_RAB";
	_mkdir(m_strFoldName.c_str());

	// 初期化
	Init_RAB();
	m_vecHist		= vector<vector<vector<double>>>(CLASS_NUM);
	m_vecHistDtNum	= vector<vector<vector<int>>>(CLASS_NUM);
	// データ重みの初期化
	WeightIniOld();
	//WeightIni();
	// 確率密度分布のBIN幅を作成
	CompSpltPnt();
	// 特徴強度を求める
	CreateFtrStrngth();
	// 学習
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;
		
		#pragma omp parallel for
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// 弱識別器の選択
			SelectClassifier(nRound, nClass);
			// データ重みの更新
			DataWeightUpdate(nRound, nClass);
			// 分類スコアの更新
			UpdateClScore(nRound, nClass);
		}
		// 最大分類スコアによる識別結果
		UpdateClResult();
		// 閾値による識別結果
		UpdateClResult(ThrScore_RAB);
		// csvファイルにエラー率を出力
		OutCsvError(nRound);
		// csvファイルに識別器を出力
		OutCsvClssfr(nRound);
		// 確率密度分布を出力
		OutCsvHist(nRound);
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

void CRealAdaBoost::Test()
{
	m_strFoldName = "Test_RAB";
	_mkdir(m_strFoldName.c_str());

	// 初期化
	Init_RAB();
	// 識別器の読み込み
	ReadClassifier();
	// 特徴強度を求める
	CreateFtrStrngth();
	// Test
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;
		
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// 分類スコアの更新
			UpdateClScore(nRound, nClass);
		}
		// 最大分類スコアによる識別結果
		UpdateClResult();
		// 閾値による識別結果
		UpdateClResult(ThrScore_RAB);
		// csvファイルにエラー率を出力
		OutCsvError(nRound);
		// csvファイルに識別器を出力
		OutCsvClssfrOld(nRound);
	}
	// 識別器をdatファイルに出力
	OutDatClssfr();
	// 識別の詳細を出力
	OutCsvDetError();
	// 特徴量を出力
	OutCsvFeature();
	// Recall Precision Curve
	DrawCurve();
	// 分類における特徴量の寄与率を計算
	AnalyzeFtrScr();
}

void CRealAdaBoost::ReadClassifier()
{
	int nDimNum = GetDimNum();

	vector<vector<float>> vecSpltPnt(nDimNum, vector<float>(BIN[0]+BIN[1]));

	//弱識別器の読み込み
	fstream fsReadCl;
	fsReadCl.open("Training_RAB\\Classifier_RAB.dat", ios::in | ios::binary);
	if (fsReadCl.is_open()){
		vector<int>		HistDim(HIST_DIM);
		vector<double>	Lut(PREDICT_NUM);
		
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				//読み込み
				fsReadCl.read((char*)&HistDim[0], sizeof(int)*HIST_DIM);
				fsReadCl.read((char*)&Lut[0], sizeof(double)*PREDICT_NUM);
				//保存
				m_vecWeakCl[nRound*CLASS_NUM + nClass].HistDim	= HistDim;
				m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut		= Lut;
			}
		}
		fsReadCl.close();
	}

	//BINの読み込み
	fstream fsReadBin;
	fsReadBin.open("Training_RAB\\Bin_RAB.dat", ios::in | ios::binary);
	if (fsReadBin.is_open()){
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsReadBin.read((char*)&vecSpltPnt[nDim][0], sizeof(float)*(BIN[0]+BIN[1]));
		fsReadBin.close();
	}
	// 保存
	m_vecSpltPnt.swap(vecSpltPnt);
}

void CRealAdaBoost::OutCsvClssfrOld(int nRound)
{
	Classifier* lpWeakCl;

	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Classifier_RAB.csv", ios::out | ios::app);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Round" << nRound << endl;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;
			fsOutCsv << "HistDim, , Lut" << endl;

			lpWeakCl = GetWeakCl(nRound, nClass);
			for(int i = 0; i < HIST_DIM; ++i)
				fsOutCsv << lpWeakCl->HistDim[i] << ",";
			for(int j = 0; j < PREDICT_NUM; ++j)
				fsOutCsv << lpWeakCl->Lut[j] << ",";
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}
}

void CRealAdaBoost::OutCsvClssfr(int nRound)
{
	Classifier* lpWeakCl;

	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Classifier_RAB.csv", ios::out | ios::app);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Round," << nRound << endl;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class," << nClass << "\nHistDim";
			// 情報取得
			lpWeakCl = GetWeakCl(nRound, nClass);
			int i;
			// 特徴次元
			for(i = 0; i < HIST_DIM; ++i)
				fsOutCsv << "," << lpWeakCl->HistDim[i];
			fsOutCsv << "\nPosDataNum";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHistDtNum[nClass][i][POS];
			fsOutCsv << "\nPosWeight";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHist[nClass][i][POS];
			fsOutCsv << "\nPosW-NegW";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHist[nClass][i][POS]-m_vecHist[nClass][i][NEG];
			fsOutCsv << "\nLut";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << lpWeakCl->Lut[i];
			fsOutCsv << "\nPosW+NegW";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHist[nClass][i][POS]+m_vecHist[nClass][i][NEG];
			fsOutCsv << "\nNegWeight";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHist[nClass][i][NEG];
			fsOutCsv << "\nNegDataNum";
			for(i = 0; i < PREDICT_NUM; ++i)
				fsOutCsv << "," << m_vecHistDtNum[nClass][i][NEG];
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}
}

void CRealAdaBoost::OutDatClssfr()
{
	int nDimNum = GetDimNum();

	fstream fsOutDatCl;
	fsOutDatCl.open(GetFoldName() + "\\Classifier_RAB.dat", ios::out | ios::binary);
	if (fsOutDatCl.is_open())
	{
		Classifier*		lpWeakCl;
		vector<int>		HistDim;
		vector<double>	Lut;
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
			{	
				lpWeakCl = GetWeakCl(nRound, nClass);
				
				HistDim = lpWeakCl->HistDim;
				Lut		= lpWeakCl->Lut;

				//出力
				fsOutDatCl.write((const char*)&HistDim[0], sizeof(int)*HIST_DIM);
				fsOutDatCl.write((const char*)&Lut[0], sizeof(double)*PREDICT_NUM);
			}
		}
		fsOutDatCl.close();
	}

	//BINを出力
	fstream fsOutDatBin;
	fsOutDatBin.open(GetFoldName() + "\\Bin_RAB.dat", ios::out | ios::binary);
	if (fsOutDatBin.is_open()){
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsOutDatBin.write((const char*)&m_vecSpltPnt[nDim][0], sizeof(float)*(BIN[0]+BIN[1]));
		fsOutDatBin.close();
	}
}

void CRealAdaBoost::OutCsvHist(int nRound)
{
	fstream fsOutCsv;

	fsOutCsv.open(GetFoldName() + "\\Histogram.csv", ios::out | ios::app);
	if (fsOutCsv.is_open()){

		fsOutCsv << "Round" << nRound << endl;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << "\nPOS" << endl;
			for(int i = 0; i < BIN[1]; ++i){
				for(int j = 0; j < BIN[0]; ++j)
					fsOutCsv << m_vecHist[nClass][i*BIN[0]+j][POS] << ",";
				fsOutCsv << endl;
			}
			fsOutCsv << "NEG" << endl;
			for(int i = 0; i < BIN[1]; ++i){
				for(int j = 0; j < BIN[0]; ++j)
					fsOutCsv << m_vecHist[nClass][i*BIN[0]+j][NEG] << ",";
				fsOutCsv << endl;
			}
		}
		fsOutCsv.close();
	}
}

void CRealAdaBoost::AnalyzeFtrScr()
{
	int				nDataNum		= GetDataNum();
	int				nDimNum			= GetDimNum();
	vector<int>&	vecMxScrClss	= GetMxScrClss();
	vector<int>&	vecLabel		= GetLabel();

	vector<vector<double>>	vecFtrScr(nDataNum, vector<double>(nDimNum, 0.0));
	vector<vector<double>>	vecNrmFtrScr(nDataNum, vector<double>(nDimNum, 0.0));
	Classifier*				lpWeakCl;
	double					dScore, dSumScr;
	int						nHistDim;
	int						tmp;
	double					dMinScr, dSumNrmScr;

	for(int nData = 0; nData < nDataNum; ++nData){
		// 初期化
		dSumScr = 0.0;
		// 識別クラス
		tmp = vecMxScrClss[nData];
		if(tmp == -1)
			continue;
		// 識別スコアを累積
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			// スコア
			dScore = Prediction(nRound, tmp, nData);
			// 弱識別器
			lpWeakCl = GetWeakCl(nRound, tmp);
			for(int i = 0; i < HIST_DIM; ++i){
				// 特徴次元
				nHistDim = lpWeakCl->HistDim[i];
				// 累積
				vecFtrScr[nData][nHistDim] += dScore;
				// 総スコア
				dSumScr += dScore;
			}
		}
		// 最小値
		dMinScr = DBL_MAX;
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			if(vecFtrScr[nData][nDim] < dMinScr)
				dMinScr = vecFtrScr[nData][nDim];
		}

		// スコア - 最小値
		dSumNrmScr = 0.0;
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			vecNrmFtrScr[nData][nDim] = vecFtrScr[nData][nDim] - dMinScr;
			dSumNrmScr += vecNrmFtrScr[nData][nDim];
		}
		if(dSumNrmScr == 0.0)
			continue;
		// 正規化
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			vecNrmFtrScr[nData][nDim] /= dSumNrmScr;

		if(dSumScr == 0.0)
			continue;
		// 正規化
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			vecFtrScr[nData][nDim] /= dSumScr;
	}

	// Csvに出力
	fstream fsOutCsvNrm;
	fsOutCsvNrm.open(GetFoldName() + "\\NrmFeatureScore.csv", ios::out);
	if (fsOutCsvNrm.is_open()){
		// 特徴次元を出力
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsOutCsvNrm << nDim << ",";
		fsOutCsvNrm << endl;
		// ラベル
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			// 識別クラス
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){				
				// nLabel->nClassに分類されたデータ
				for(int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData] != nLabel || vecMxScrClss[nData] != nClass)
						continue;
					// 出力
					fsOutCsvNrm << nLabel << " -> " << nClass << endl;
					for(int nDim = 0; nDim < nDimNum; ++nDim)
						fsOutCsvNrm << vecNrmFtrScr[nData][nDim] << ",";
					fsOutCsvNrm << endl;
				}
			}
		}
		fsOutCsvNrm.close();
	}

	// Csvに出力
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\FeatureScore.csv", ios::out);
	if (fsOutCsv.is_open()){
		// 特徴次元を出力
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsOutCsv << nDim << ",";
		fsOutCsv << endl;
		// ラベル
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			// 識別クラス
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				// 初期化
				vector<double> vecSumFtrScr(nDimNum, 0.0);
				// nLabel->nClassに分類されたデータ
				for(int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData] != nLabel || vecMxScrClss[nData] != nClass)
						continue;
					for(int nDim = 0; nDim < nDimNum; ++nDim)
						vecSumFtrScr[nDim] += vecNrmFtrScr[nData][nDim];
						//vecSumFtrScr[nDim] += vecFtrScr[nData][nDim];
				}
				// 出力
				fsOutCsv << nLabel << " -> " << nClass << endl;
				for(int nDim = 0; nDim < nDimNum; ++nDim)
					fsOutCsv << vecSumFtrScr[nDim] << ",";
				fsOutCsv << endl;
			}
		}
		fsOutCsv.close();
	}

	// 保存
	//m_vecFtrScr.swap(vecFtrScr);
	m_vecFtrScr.swap(vecNrmFtrScr);
}

/*
void CRealAdaBoost::ReadParameter()
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

void CRealAdaBoost::ScoreToProb()
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

void CRealAdaBoost::ProbTest()
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

	//エラー詳細を書き出し
	OutCsvDetError();

	cout << "<Segment Classification (Using Probability)>" << endl;
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		cout << "--Class" << nClass << "--" << endl;
		cout << " Label:" << m_LblNum[nClass] << " Class:" << m_ClssNum[nClass] << " TruePositive:" << m_TrPsNum[nClass] << endl;
		cout << fixed << setprecision(8) << " Recall:" << (double)m_TrPsNum[nClass] / (double)m_LblNum[nClass] << " Precision:" << (double)m_TrPsNum[nClass] / (double)m_ClssNum[nClass] << endl;
	}
	
	return;
}
*/