#pragma once

#include "learning.h"

using namespace std;

void CLearning::InitErrFile(string& strFileName)
{
	fstream	fsOutCsv;
	fsOutCsv.open(strFileName, ios::out);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Round,";
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
			fsOutCsv << "Class" << nClass << ",Label,Class,TruePositive,Recall,Precision,";
		fsOutCsv << endl;
	}
}

void CLearning::Initialize(vector<vector<float>>& vecFeature, vector<int>& vecLabel)
{
	int	nDataNum	= vecLabel.size();
	int nDimNum		= vecFeature.size();
	
	// 各クラスのラベル数
	vector<int> vecLblNum(CLASS_NUM, 0);
	int tmp;
	for(int nData = 0; nData < nDataNum; ++nData){
		tmp = vecLabel[nData];
		++vecLblNum[tmp];
	}
	cout << "<LabelNum>\n";
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		cout << " Class " << nClass << " " << vecLblNum[nClass];
	cout << endl;

	// エラー率出力ファイルの初期化
	string strFileName = GetFoldName() + "\\ErrorRate - MaxScoreClass.csv";
	InitErrFile(strFileName);
	// エラー率出力ファイルの初期化
	strFileName = GetFoldName() + "\\ErrorRate - ThrPositiveClass.csv";
	InitErrFile(strFileName);

	m_vecScore = vector<vector<double>>(CLASS_NUM, vector<double>(nDataNum, 0.0));

	//保存
	m_nDataNum	= nDataNum;
	m_nDimNum	= nDimNum;
	m_vecFeature.swap(vecFeature);
	m_vecLabel.swap(vecLabel);
	m_vecLblNum.swap(vecLblNum);
}

void CLearning::WeightIniOld()
{
	int nDataNum = GetDataNum();

	// (データ重み) = 1.0 / (全データ数)
	double dWeight	= 1.0 / (double)nDataNum;
	m_vecWeight = vector<vector<double>>(CLASS_NUM, vector<double>(nDataNum, dWeight));
}

void CLearning::WeightIni()
{
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<int>&	vecLblNum	= GetLblNum();

	// (各クラスのデータ重み) = 1.0/(各クラスデータ数)
	vector<double> vecDtWeight(CLASS_NUM);
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		vecDtWeight[nClass] = 1.0/(double)vecLblNum[nClass];
	
	// データ重み
	vector<double> vecWeight(nDataNum);
	int tmp;
	for(int nData = 0; nData < nDataNum; ++nData){
		tmp = vecLabel[nData];
		vecWeight[nData] = vecDtWeight[tmp];
	}

	// コピー
	m_vecWeight = vector<vector<double>>(CLASS_NUM);
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		m_vecWeight[nClass] = vecWeight;
}

void CLearning::UpdateClResult()
{
	// 情報取得
	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>& vecScore = GetScore();

	vector<int> vecMxScrClss(nDataNum);				// 識別クラス
	vector<int>	vecMxScrClNum(CLASS_NUM, 0);		// 検出数
	vector<int>	vecMxScrTpNum(CLASS_NUM, 0);		// TruPositive数

	double	dMaxScore;
	int		tmp;
	for(int nData = 0; nData < nDataNum; ++nData){
		dMaxScore = -DBL_MAX;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			if(vecScore[nClass][nData] > dMaxScore){
				tmp = nClass;
				dMaxScore = vecScore[nClass][nData];
			}else if(vecScore[nClass][nData] == dMaxScore){
				tmp = -1;
			}
		}
		// 最大スコアのクラスに識別
		vecMxScrClss[nData] = tmp;
		if(tmp != -1){
			// 検出数
			++vecMxScrClNum[tmp];
			//TruePositive
			if(tmp == vecLabel[nData])
				++vecMxScrTpNum[tmp];
		}
	}

	// 保存
	m_vecMxScrClss.swap(vecMxScrClss);
	m_vecMxScrClNum.swap(vecMxScrClNum);
	m_vecMxScrTpNum.swap(vecMxScrTpNum);
}

void CLearning::UpdateClResult(const double* lpThrScore)
{
	// 情報取得
	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>& vecScore = GetScore();

	vector<vector<bool>>	vecThrPos(CLASS_NUM, vector<bool>(nDataNum, false));
	vector<int>				vecThrPsNum(CLASS_NUM, 0);		// 各クラス検出数
	vector<int>				vecThrPsTpNum(CLASS_NUM, 0);	// 各クラスTruPositive数

	for(int nData = 0; nData < nDataNum; ++nData){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// 分類スコアが閾値以上であれば検出
			if(vecScore[nClass][nData] > lpThrScore[nClass]){
				vecThrPos[nClass][nData] = true;
				// 検出数
				++vecThrPsNum[nClass];
				// TruePositive
				if(vecLabel[nData] == nClass)
					++vecThrPsTpNum[nClass];
			}
		}
	}

	// 保存
	m_vecThrPos.swap(vecThrPos);
	m_vecThrPsNum.swap(vecThrPsNum);
	m_vecThrPsTpNum.swap(vecThrPsTpNum);
}

void CLearning::OutputErr(string& strFileName, vector<int>& vecClssNum, vector<int>& vecTrPsNum, int nRound)
{
	vector<int>& vecLblNum = GetLblNum();

	fstream fsOutCsv;
	fsOutCsv.open(strFileName, ios::out | ios::app);
	if (fsOutCsv.is_open()){
		double dRcll;	//再現率
		double dPrcsn;	//適合率
		fsOutCsv << nRound << ",";
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			dRcll	= (double)vecTrPsNum[nClass] / (double)vecLblNum[nClass];
			dPrcsn	= (double)vecTrPsNum[nClass] / (double)vecClssNum[nClass];
			//出力
			fsOutCsv << "," << vecLblNum[nClass] << "," << vecClssNum[nClass] << "," << vecTrPsNum[nClass] << "," << dRcll << "," << dPrcsn << ",";
		}
		fsOutCsv << endl;
	}
}

void CLearning::OutCsvError(int nRound)
{
	string strFileName = GetFoldName() + "\\ErrorRate - MaxScoreClass.csv";
	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);

	strFileName = GetFoldName() + "\\ErrorRate - ThrPositiveClass.csv";
	OutputErr(strFileName, m_vecThrPsNum, m_vecThrPsTpNum, nRound);
}

void CLearning::OutCsvDetError()
{
	int						nDataNum		= GetDataNum();
	vector<int>&			vecLabel		= GetLabel();
	vector<int>&			vecMxScrClss	= GetMxScrClss();
	vector<vector<bool>>&	vecThrPos		= GetThrPos();

	//int c;
	//fstream fsOut;
	//fsOut.open("DetailError - MaxScoreClass_update.csv", ios::out);
	//if (fsOut.is_open()){
	//	for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
	//			c = 0;
	//			for (int nData = 0; nData < nDataNum; ++nData){
	//				if(vecLabel[nData]==nLabel && vecMxScrClss[nData]==-1)
	//					++c;
	//			}
	//			fsOut << nLabel << "->" << "-1" << "," << c <<  endl;
	//	}
	//	fsOut.close();
	//}
 

	int Count; 
	fstream fsOutCsvDetailOld;
	fsOutCsvDetailOld.open(GetFoldName() + "\\DetailError - MaxScoreClass.csv", ios::out);
	if (fsOutCsvDetailOld.is_open()){
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				Count = 0;
				for (int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData]==nLabel && vecMxScrClss[nData]==nClass)
						++Count;
				}
				fsOutCsvDetailOld << nLabel << "->" << nClass << "," << Count <<  endl;
			}
		}
		fsOutCsvDetailOld.close();
	}

	fstream fsOutCsvDetail;
	fsOutCsvDetail.open(GetFoldName() + "\\DetailError - ThrPositiveClass.csv", ios::out);
	if (fsOutCsvDetail.is_open()){
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				Count = 0;
				for (int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData] == nLabel && vecThrPos[nClass][nData])
						++Count;
				}
				fsOutCsvDetail << nLabel << "->" << nClass << "," << Count <<  endl;
			}
		}
		fsOutCsvDetail.close();
	}
}

void CLearning::OutCsvFeature()
{
	int						nDataNum		= GetDataNum();
	int						nDimNum			= GetDimNum();
	vector<int>&			vecLabel		= GetLabel();
	vector<int>&			vecMxScrClss	= GetMxScrClss();
	vector<vector<double>>& vecScore		= GetScore();
	vector<vector<float>>&	vecFeature		= GetFeature();

	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Feature.csv", ios::out);
	if (fsOutCsv.is_open()){
		//クラスごとに出力
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;
			//特徴量を出力
			for(int nDim = 0; nDim < nDimNum; ++nDim){
				fsOutCsv << "DIM" << nDim << endl;
				for(int nData = 0 ; nData < nDataNum; ++nData){
					if(vecLabel[nData] == nClass)
						fsOutCsv << vecFeature[nDim][nData] << ",";
				}
				fsOutCsv << endl;
			}
			fsOutCsv << endl;
			//分類スコアを出力
			for(int i = 0; i < CLASS_NUM; ++i){
				fsOutCsv << "Score" << i <<  endl;
				for(int nData = 0 ; nData < nDataNum; ++nData){
					if(vecLabel[nData] == nClass)
						fsOutCsv << vecScore[i][nData] << ",";
				}
				fsOutCsv << endl;
			}
			fsOutCsv << endl;
			//識別結果を出力
			fsOutCsv << "Predict" << nClass << endl;
			for(int nData = 0 ; nData < nDataNum; ++nData){
				if(vecLabel[nData] == nClass)
					fsOutCsv << vecMxScrClss[nData] << ",";
			}	
			fsOutCsv << "\n" << endl;
		}
		fsOutCsv.close();
	}
}

void CLearning::DrawCurve()
{
	int						nDataNum	= GetDataNum();
	vector<int>&			vecLabel	= GetLabel();
	vector<int>&			vecLblNum	= GetLblNum();
	vector<vector<double>>& vecScore	= GetScore();

	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\Recall-Precision Curve.csv", ios::out);
	if (fsOutCsv.is_open()){
		fsOutCsv << "Threshold, Recall, Precision" << endl;
		//分類スコアの最大・最小値
		vector<double> vecMaxScr(CLASS_NUM, -DBL_MAX);
		vector<double> vecMinScr(CLASS_NUM, DBL_MAX);
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			for(int nData = 0; nData < nDataNum; ++nData){
				if(vecScore[nClass][nData] > vecMaxScr[nClass])
					vecMaxScr[nClass] = vecScore[nClass][nData];
				if(vecScore[nClass][nData] < vecMinScr[nClass])
					vecMinScr[nClass] = vecScore[nClass][nData];
			}
		}
		//各スコア閾値におけるRecallとPrecision
		double	dPitch = 0.4;
		double	dVal;
		double	dRcll, dPrcsn, dRcllOld, dPrcsnOld;
		int		nLblNum, nClssNum, nTrPsNum;
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;

			nLblNum = vecLblNum[nClass];
			dVal	= vecMinScr[nClass]-dPitch;

			int cnt = 1;
			while(dVal < vecMaxScr[nClass]){
				nClssNum = 0;
				nTrPsNum = 0;
				for(int nData = 0; nData < nDataNum; ++nData){
					if(vecScore[nClass][nData] > dVal){
						//検出数
						++nClssNum;
						//TruePositive
						if(vecLabel[nData] == nClass)
							++nTrPsNum;
					}
				}
				//Recall, Precision
				dRcll	= (double)nTrPsNum / (double)nLblNum;
				dPrcsn	= (double)nTrPsNum / (double)nClssNum;

				if(dRcll == dRcllOld && dPrcsn == dPrcsnOld){
					++cnt;
				}else{
					//出力
					fsOutCsv << dVal << "," << dRcll << "," << dPrcsn << endl;
				}
				if(cnt > 1000)
					break;

				dRcllOld	=	dRcll;
				dPrcsnOld	=	dPrcsn;
				dVal		+=	dPitch*(double)cnt;
			}
		}
		fsOutCsv.close();
	}
}

void CLearning::AnalyzeClProb()
{
	//クラス確率表の作成
	CompProbTbl();
	//ロジスティック回帰分析
	LogRegression();
}

void CLearning::CompProbTbl()
{
	m_vecScrTbl	= vector<vector<double>>(CLASS_NUM);
	m_vecPrbTbl	= vector<vector<double>>(CLASS_NUM);

	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>&	vecScore = GetScore();

	//分類スコアを昇順に変換
	vector<vector<double>> vecSortScr = m_vecScore;
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		sort(vecSortScr[nClass].begin(), vecSortScr[nClass].end());

	int nSpltPnt = nDataNum / SCORE_BIN;

	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		// 初期化
		vector<double>	vecScrBin(SCORE_BIN);
		vector<double>	vecPrbTbl(SCORE_BIN);
		vector<int>		vecCount(SCORE_BIN, 0);
		int nSpltNum = 0;
		// BINの作成
		for(int i = 0; i < SCORE_BIN; ++i){
			nSpltNum += nSpltPnt;
			if(nSpltNum > nDataNum-1)
				nSpltNum = nDataNum - 1;
			vecScrBin[i] = vecSortScr[nClass][nSpltNum];
		}
		// 各BINのデータ数をカウント
		size_t stBin;
		vector<double>::iterator itr;
		for(int nData = 0; nData < nDataNum; ++nData){
			if(vecLabel[nData] != nClass)
				continue;
			// 二分探索
			itr		= upper_bound(vecScrBin.begin(), vecScrBin.end(), vecScore[nClass][nData]);
			stBin	= distance(vecScrBin.begin(), itr);
			if(stBin > SCORE_BIN-1)
				stBin = SCORE_BIN-1;
			++vecCount[stBin];
		}
		// クラス確率表
		for(int i = 0; i < SCORE_BIN; ++i)
			vecPrbTbl[i] = (double)vecCount[i]/(double)nSpltPnt;
		// 保存
		m_vecScrTbl[nClass].swap(vecScrBin);
		m_vecPrbTbl[nClass].swap(vecPrbTbl);
	}

	// csvファイルに出力
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\ProbTable.csv", ios::out);
	if (fsOutCsv.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++ nClass){
			fsOutCsv << "Class" << nClass << endl;
			fsOutCsv << "Score" << endl;
			// 出力
			for(int i = 0; i < SCORE_BIN; ++i)
				fsOutCsv << m_vecScrTbl[nClass][i] << ",";
			fsOutCsv << endl;
			// 出力
			fsOutCsv << "Probability" << endl;
			for(int i = 0; i < SCORE_BIN; ++i)
				fsOutCsv << m_vecPrbTbl[nClass][i] << ",";
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}

	//datファイルに出力
	fstream fsOutDat;
	fsOutDat.open(GetFoldName() + "\\ProbTable.dat", ios::out | ios::binary);
	if (fsOutDat.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			//出力
			fsOutDat.write((const char*)&m_vecScrTbl[nClass][0], sizeof(double) * SCORE_BIN);
			fsOutDat.write((const char*)&m_vecPrbTbl[nClass][0], sizeof(double) * SCORE_BIN);
		}
		fsOutDat.close();
	}
}

void CLearning::LogRegression()
{
	m_vecSgmdPrm = vector<vector<double>>(CLASS_NUM, vector<double>(2));
	
	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>& vecScore = GetScore();

	//（仮）
	vector<double>	pi(nDataNum);
	double			a, Lw, DrvLw, Gradw0, Gradw1;
	double eta		= 0.00001;
	double epsilon	= 0.00000001;

	// 計算&csvファイルに出力
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\LogisticRegression.csv", ios::out);
	if (fsOutCsv.is_open()){

		for(int nClass = 0; nClass < CLASS_NUM; ++ nClass){
			cout << "\nClass" << nClass << endl;
			fsOutCsv << "\nClass" << nClass << "\nLw, DrvLw, w0, w1" << endl;

			// 教師データの作成
			vector<int> t(nDataNum, 0);
			for(int nData = 0; nData < nDataNum; ++nData){
				if(vecLabel[nData] == nClass){
					t[nData] = 1;
				}
			}
			//（仮）
			int		cnt = 0;
			double	w0	= -2.0;
			double	w1	= 1.5;
			double tmp;
			do{
				Lw = 0.0;
				for(int nData = 0; nData < nDataNum; ++nData){
					a = w0 + w1*vecScore[nClass][nData];
					tmp = exp(a);
					pi[nData] = tmp / (1.0+tmp);
//					Lw += t[nData] * log(pi[nData]) + (1-t[nData]) * log(1-pi[nData]);
					Lw += t[nData]*a - log(1.0+tmp);
				}
				Lw *= -1.0;

				Gradw0	= 0.0;
				Gradw1	= 0.0;
				DrvLw	= 0.0;
				for(int nData = 0; nData < nDataNum; ++nData){
					Gradw0	+= pi[nData] - t[nData];
					Gradw1	+= vecScore[nClass][nData] * (pi[nData]-t[nData]);
					DrvLw	+= (pi[nData]-t[nData]) + (vecScore[nClass][nData] * (pi[nData]-t[nData]));
				}
				w0 -= eta * Gradw0;
				w1 -= eta * Gradw1;
				
				if(cnt != 0){
					fsOutCsv << Lw << "," << DrvLw << "," << w0 << "," << w1 << endl;
				}
				++cnt; 

			}while(fabs(DrvLw) > epsilon);

			cout << "w0:" << fixed << setprecision(16) << w0 << endl;
			cout << "w1:" << fixed << setprecision(16) << w1 << endl;

			// 保存
			m_vecSgmdPrm[nClass][0] = w0;
			m_vecSgmdPrm[nClass][1] = w1;
		}
		fsOutCsv.close();
	}

	//datファイルに出力
	fstream fsOutDat;
	fsOutDat.open(GetFoldName() + "\\SigmoidParam.dat", ios::out | ios::binary);
	if (fsOutDat.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
			fsOutDat.write((const char*)&m_vecSgmdPrm[nClass][0], sizeof(double)*2);
		fsOutDat.close();
	}
}

void CLearning::ReadSgmdParam()
{
	// シグモイド関数のパラメータ
	m_vecSgmdPrm = vector<vector<double>>(CLASS_NUM, vector<double>(2));

	fstream fsRead;
	fsRead.open(GetFoldName() + "\\SigmoidParam.dat", ios::in | ios::binary);
	if (fsRead.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
			fsRead.read((char*)&m_vecSgmdPrm[nClass][0], sizeof(double)*2);
		fsRead.close();
	}
}

void CLearning::Score2Prob(vector<vector<double>>& vecScore, vector<vector<double>>& vecClProb)
{
	vector<vector<double>>& vecSgmdParam = GetSgmdParam();
	int nDataNum = GetDataNum();

	vector<vector<double>> vecProb(CLASS_NUM, vector<double>(nDataNum));

	//シグモイド関数より確率に変換
	double dAlpha;
	for(int nData = 0; nData < nDataNum; ++nData){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			dAlpha = exp(vecSgmdParam[nClass][0] + vecScore[nClass][nData]*vecSgmdParam[nClass][1]);
			vecProb[nClass][nData] = dAlpha / (1.0 + dAlpha);
		}
	}

	vecClProb.swap(vecProb);

	// 情報登録
//	m_vecScore = vecClProb;
//	// 最大分類スコアによる識別結果
//	UpdateClResult();
//	// 識別結果を出力
//	int nRound = LEARN_NUM-1;
//	string strFileName = "ErrorRate - ClassProbabilityClassify.csv";
//	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);
}

void CLearning::TimeSeriesClassify(vector<vector<double>>& vecLogOdds)
{
	// 情報登録
	m_vecScore.swap(vecLogOdds);

	// 最大分類スコアによる識別結果
	UpdateClResult();

	// 識別結果を出力
	int nRound = LEARN_NUM-1;
	string strFileName = GetFoldName() + "\\ErrorRate - TimeSeriesClassify.csv";
	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);

	// 識別の詳細を出力
	OutCsvDetError();

	// Recall Precision Curve
	DrawCurve();
}

void CLearning::AnalyzeRcPrCurve(vector<int>& vecLimitLbl, vector<vector<double>>& vecLimitScr)
{
	int nDataNum = vecLimitLbl.size();

	// 各クラスのラベル数
	vector<int> vecLblNum(CLASS_NUM, 0);
	int tmp;
	for(int nData = 0; nData < nDataNum; ++nData){
		tmp = vecLimitLbl[nData];
		++vecLblNum[tmp];
	}

	m_nDataNum = nDataNum;
	m_vecLabel.swap(vecLimitLbl);
	m_vecScore.swap(vecLimitScr);
	m_vecLblNum.swap(vecLblNum);

	// 最大分類スコアによる識別結果
	UpdateClResult();

	// 識別結果を出力
	int nRound = LEARN_NUM-1;
	string strFileName = GetFoldName() + "\\ErrorRate - LimitedData.csv";
	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);

	// Recall Precision Curve
	DrawCurve();
}

void CLearning::WriteClResult(string strFileName, vector<int>& vecClass)
{
	int nDataNum = GetDataNum();

	fstream fsOutDat;
	fsOutDat.open(strFileName, ios::out | ios::binary);
	if (fsOutDat.is_open()){
		fsOutDat.write((const char*)&vecClass[0], sizeof(int)*nDataNum);
		fsOutDat.close();
	}
}

void CLearning::ReadClResult(string strFileName, vector<int>& vecClass)
{
	int nDataNum = GetDataNum();
	vector<int> vecReadClss(nDataNum);

	fstream fsRead;
	fsRead.open(strFileName, ios::in | ios::binary);
	if (fsRead.is_open()){
		fsRead.read((char*)&vecReadClss[0], sizeof(int)*nDataNum);
		fsRead.close();
	}

	vecClass.swap(vecReadClss);
}

string CLearning::GetCrrntTime()
{
	// 現在時刻
	time_t	t	= time(NULL);
    tm*		ptr	= localtime(&t);
	stringstream ssTime;
	ssTime << ptr->tm_hour << ":" << ptr->tm_min << ":" << ptr->tm_sec;

	return ssTime.str();
}