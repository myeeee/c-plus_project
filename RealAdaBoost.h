#pragma once

#include "learning.h"

#define	HIST_DIM	2				//確率密度分布の次元数

//const int	BIN[HIST_DIM]	= {64, 32};
const int	BIN[HIST_DIM]	= {64, 20};
const int	PREDICT_NUM		= BIN[0]*BIN[1];

const double ThrScore_RAB[CLASS_NUM] = {0.0, 0.0, 0.0, 0.0};	// 分類スコアの閾値

using namespace std;

class CRealAdaBoost : public CLearning
{
public:
	// 識別器(RAB)
	struct Classifier{
		vector<int>		HistDim;	//特徴次元
		vector<double>	Lut;		//確率密度分布
		//コンストラクタ
		Classifier(){
			Lut = vector<double>(PREDICT_NUM);
		}
	};

private:
	// メンバ変数
	vector<Classifier>				m_vecWeakCl;
	vector<vector<float>>			m_vecSpltPnt;
	vector<vector<vector<double>>>	m_vecHist;
	vector<vector<vector<int>>>		m_vecHistDtNum;
	vector<vector<int>>				m_vecBinNum;
	vector<vector<double>>			m_vecFtrScr;

public:
	// メンバ関数
	void	Training();
	void	Test();

	// 内部使用関数
	void	Init_RAB();
	void	CompSpltPnt();
	void	SelectClassifier(int nRound, int nClass);
	void	CompHist(vector<vector<double>>& Lut, vector<int>& vecDim, int nClass);	
	void	CompHist(vector<vector<double>>& Lut, vector<int>& vecDim, int nClass, vector<vector<int>>& vecHistDtNum);
	int		CompBinOld(vector<int>& vecDim, int nData);
	int		CompBin(vector<int>& vecDim, int nData);
	void	SmoothingLUT1(vector<vector<double>>& Lut);
	void	SmoothingLUT2(vector<vector<double>>& Lut);
	double	EvaluateHist(vector<vector<double>>& lut);
	void	DataWeightUpdate(int nRound, int nClass);
	double	Prediction(int nRound, int nClass, int nData);
	void	UpdateClScore(int nRound, int nClass);
	void	OutCsvClssfr(int nRound);
	void	OutCsvClssfrOld(int nRound);
	void	OutDatClssfr();
	void	OutCsvHist(int nRound);
	void	ReadClassifier();
	void	CreateFtrStrngth();
	void	AnalyzeFtrScr();

	// 情報取得
	Classifier*	GetWeakCl(int nRound, int nClass){	return	&m_vecWeakCl[nRound*CLASS_NUM + nClass];	};
	vector<vector<float>>&	GetSpltPnt(){	return	m_vecSpltPnt;	};
	vector<vector<double>>&	GetFtrScr(){	return	m_vecFtrScr;	};

/*	
	void ReadParameter();
	void ScoreToProb();
	void ProbTest();
	vector<vector<double>>&	GetSgmdProb(){ return m_SgmdProb; };
	vector<vector<double>>& GetTblProb(){ return m_TblProb;	};
*/
};
/*
inline void CRealAdaBoost::SelectClassifier(int nRound, int nClass)
{
	int		nDimNum = GetDimNum();
	double	epsilon	= 1.0 / double(GetLblNum(nClass));
	
	// 特徴次元
	vector<int> vecDim(HIST_DIM);
	vector<int> vecHistDim;
	// 確率密度分布
	vector<vector<double>> lut;
	vector<vector<double>> Lut;
	// 評価値
	double dValue;
	double dMinValue = DBL_MAX;

	for(int nDim0 = 0; nDim0 < nDimNum; ++nDim0){
		for(int nDim1 = 0; nDim1 < nDimNum; ++nDim1){
			if(nDim0 == nDim1)
				continue;
			vecDim[0] = nDim0;
			vecDim[1] = nDim1;
			// 確率密度分布の作成
			CompHist(lut, vecDim, nClass);
			// 評価値の算出
			dValue = EvaluateHist(lut);
			if(dValue < dMinValue){
				vecHistDim	= vecDim;
				Lut			= lut;
				dMinValue	= dValue;
			}
		}
	}
	// 識別器の保存
	m_vecWeakCl[nRound*CLASS_NUM + nClass].HistDim = vecHistDim;
	// 飽和関数を使用（仮）
	double n		= 0.0;
	double theta	= 0.8;
	double tmp		= pow(10.0, theta);
	double data_num	= 1.0 / (double)m_nDataNum;
	double a		= pow(10.0, n);
	double b		= pow(data_num, tmp);
	double e		= pow(b, a);
	double PosW, NegW;
	for(int i = 0; i < PREDICT_NUM; ++i){
//		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = 0.5 * log((Lut[i][POS]+epsilon) / (Lut[i][NEG]+epsilon));
//		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = (Lut[i][POS]-Lut[i][NEG]) / (Lut[i][POS]+Lut[i][NEG]+epsilon);
		PosW = pow(Lut[i][POS], a);
		NegW = pow(Lut[i][NEG], a);
		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = (PosW-NegW) / (PosW+NegW+e);
	}

	// 確率密度分布の保存
	m_vecHist[nClass].swap(Lut);
}	
*/

inline void CRealAdaBoost::SelectClassifier(int nRound, int nClass)
{
	int		nDimNum = GetDimNum();
//	double	epsilon	= 1.0 / double(GetLblNum(nClass));
	double	epsilon	= 1.0 / double(GetDataNum());
	
	// 特徴次元
	vector<int> vecDim(HIST_DIM);
	vector<int> vecHistDim;
	// 確率密度分布
	vector<vector<double>> lut;
	vector<vector<double>> Lut;
	// 評価値
	double dValue;
	double dMinValue = DBL_MAX;

	vector<vector<int>> vecDtNum;
	vector<vector<int>> vecHistDtNum;

	for(int nDim0 = 0; nDim0 < nDimNum; ++nDim0){
		for(int nDim1 = 0; nDim1 < nDimNum; ++nDim1){
			if(nDim0 == nDim1)
				continue;
			vecDim[0] = nDim0;
			vecDim[1] = nDim1;
			// 確率密度分布の作成
			CompHist(lut, vecDim, nClass, vecDtNum);
			// 評価値の算出
			dValue = EvaluateHist(lut);
			if(dValue < dMinValue){
				vecHistDim		= vecDim;
				Lut				= lut;
				dMinValue		= dValue;
				vecHistDtNum	= vecDtNum;
			}
		}
	}
	// 識別器の保存
	m_vecWeakCl[nRound*CLASS_NUM + nClass].HistDim = vecHistDim;
	// 飽和関数を使用（仮）
	double n		= 0.0;
	double theta	= 0.8;
	double tmp		= pow(10.0, theta);
	double data_num	= 1.0 / (double)m_nDataNum;
	double a		= pow(10.0, n);
	double b		= pow(data_num, tmp);
	double e		= pow(b, a);
//	double p		= pow(epsilon, 10.0);
//	double p		= epsilon*50.0;
	double p		= epsilon;
	double PosW, NegW;
	for(int i = 0; i < PREDICT_NUM; ++i){
//		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = 0.5 * log((Lut[i][POS]+epsilon) / (Lut[i][NEG]+epsilon));
//		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = (Lut[i][POS]-Lut[i][NEG]) / (Lut[i][POS]+Lut[i][NEG]+epsilon);
		PosW = pow(Lut[i][POS], a);
		NegW = pow(Lut[i][NEG], a);
//		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = (PosW-NegW) / (PosW+NegW+e);
		m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] = (PosW-NegW) / (PosW+NegW+p);
//		if(vecHistDtNum[i][POS]<2 && vecHistDtNum[i][NEG]<2)
//			cout << "PosDtNum " << vecHistDtNum[i][POS] << " NegDtNum " << vecHistDtNum[i][NEG] << " PosW " << PosW << " NegW " << NegW << " Lut " << m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut[i] << " e " << e << endl;
	}

	// 確率密度分布の保存
	m_vecHist[nClass].swap(Lut);
	m_vecHistDtNum[nClass].swap(vecHistDtNum);
}	

inline void CRealAdaBoost::CompHist(vector<vector<double>>& Lut, vector<int>& vecDim, int nClass, vector<vector<int>>& vecHistDtNum)
{
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<double>&	vecWeight	= GetWeight(nClass);

	vector<vector<double>>	lut(PREDICT_NUM, vector<double>(2, 0.0));
	int nBin;
	vector<vector<int>>		vecDtNum(PREDICT_NUM, vector<int>(2, 0));

	for(int nData = 0; nData < nDataNum; ++nData){
		// BINの計算
		nBin = CompBin(vecDim, nData);
		// Positiveクラス
		if(vecLabel[nData] == nClass){
			lut[nBin][POS] += vecWeight[nData];
			++vecDtNum[nBin][POS];
		// Negativeクラス
		}else{
			lut[nBin][NEG] += vecWeight[nData];
			++vecDtNum[nBin][NEG];
		}
	}
	// 確率密度分布の平滑化
	SmoothingLUT1(lut);
	//SmoothingLUT2(lut);

	//確率密度分布の正規化
	double dSumPos = 0.0;
	double dSumNeg = 0.0;
	for(int i = 0; i < PREDICT_NUM; ++i){
		dSumPos += lut[i][POS];
		dSumNeg += lut[i][NEG];
	}
	for(int j = 0; j < PREDICT_NUM; ++j){
		lut[j][POS] = lut[j][POS] / dSumPos;
		lut[j][NEG] = lut[j][NEG] / dSumNeg;
	}

	//保存
	Lut.swap(lut);
	vecHistDtNum.swap(vecDtNum);
}

inline void CRealAdaBoost::CompHist(vector<vector<double>>& Lut, vector<int>& vecDim, int nClass)
{
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<double>&	vecWeight	= GetWeight(nClass);

	vector<vector<double>> lut(PREDICT_NUM, vector<double>(2, 0.0));
	int nBin;

	for(int nData = 0; nData < nDataNum; ++nData){
		// BINの計算
		nBin = CompBin(vecDim, nData);
		// Positiveクラス
		if(vecLabel[nData] == nClass){
			lut[nBin][POS] += vecWeight[nData];
		// Negativeクラス
		}else{
			lut[nBin][NEG] += vecWeight[nData];
		}
	}
	// 確率密度分布の平滑化
	SmoothingLUT1(lut);
	//SmoothingLUT2(lut);

	//確率密度分布の正規化
	double dSumPos = 0.0;
	double dSumNeg = 0.0;
	for(int i = 0; i < PREDICT_NUM; ++i){
		dSumPos += lut[i][POS];
		dSumNeg += lut[i][NEG];
	}
	for(int j = 0; j < PREDICT_NUM; ++j){
		lut[j][POS] = lut[j][POS] / dSumPos;
		lut[j][NEG] = lut[j][NEG] / dSumNeg;
	}

	//保存
	Lut.swap(lut);
}

inline int CRealAdaBoost::CompBin(vector<int>& vecDim, int nData)
{
	int nDimNum = GetDimNum();

	int nDim;
	int nBin[HIST_DIM];

	for(int i = 0; i < HIST_DIM; ++i){
		// 特徴次元
		nDim = vecDim[i];
		// 特徴強度
		nBin[i] = m_vecBinNum[i*nDimNum + nDim][nData];
	}
	int nBinNum = nBin[1]*BIN[0] + nBin[0];

	return	nBinNum;
}

inline int CRealAdaBoost::CompBinOld(vector<int>& vecDim, int nData)
{
	int		nDim;
	size_t	stBin[HIST_DIM];
	vector<float>::iterator itrStrt;
	vector<float>::iterator itrBound;

	for(int i = 0; i < HIST_DIM; ++i){
		nDim = vecDim[i];
		// 分割点配列のiterator
		itrStrt = m_vecSpltPnt[nDim].begin() + i*BIN[0];
		// 二分探索
		itrBound = upper_bound(itrStrt, itrStrt+BIN[i], m_vecFeature[nDim][nData]);
		// Binの番号
		stBin[i] = distance(itrStrt, itrBound);
		if(stBin[i] > BIN[i] - 1)
			stBin[i] = BIN[i] - 1;
	}
	
	int nBin = stBin[1]*BIN[0] + stBin[0];

	return nBin;
}

inline void CRealAdaBoost::SmoothingLUT1(vector<vector<double>>& Lut)
{
	//コピー
	vector<vector<double>> lut = Lut;

	double	dPos, dNeg;
	int		cnt;

	for(int i = 0; i < BIN[1]; ++i){
		for(int j = 0; j < BIN[0]; ++j){
			dPos	= 0.0;
			dNeg	= 0.0;
			cnt		= 0;
			for(int p = -1; p < 2; ++p){
				for(int q = -1; q < 2; ++q){
					if(i+p<0 || j+q<0 || i+p>=BIN[1] || j+q>=BIN[0])
						continue;

					dPos += lut[(i+p)*BIN[0] + j+q][POS];
					dNeg += lut[(i+p)*BIN[0] + j+q][NEG];
					++cnt;
				}
			}
			Lut[i*BIN[0] + j][POS] = dPos / (double)cnt;
			Lut[i*BIN[0] + j][NEG] = dNeg / (double)cnt;
		}
	}
}

inline void CRealAdaBoost::SmoothingLUT2(vector<vector<double>>& Lut)
{
	//コピー
	vector<vector<double>> lut = Lut;

	double	dPos, dNeg;
	int		cnt;
	
	for(int i = 0; i < BIN[1]; ++i){
		for(int j = 0; j < BIN[0]; ++j){
			dPos	= 0.0;
			dNeg	= 0.0;
			cnt		= 0;
			for(int q = -1; q < 2; ++q){
				if(j+q<0 || j+q>=BIN[0])
					continue;

				dPos += lut[i*BIN[0] + j+q][POS];
				dNeg += lut[i*BIN[0] + j+q][NEG];
				++cnt;
			}
			Lut[i*BIN[0] + j][POS] = dPos / (double)cnt;
			Lut[i*BIN[0] + j][NEG] = dNeg / (double)cnt;
		}
	}
}

inline double CRealAdaBoost::EvaluateHist(vector<vector<double>>& lut)
{
	double tmp;
	double dValue = 0.0;

	for(int i = 0; i < PREDICT_NUM; ++i){
		tmp = lut[i][POS] * lut[i][NEG];
		dValue += sqrt(tmp);
	}

	return dValue;
}

inline void CRealAdaBoost::DataWeightUpdate(int nRound, int nClass)
{
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<double>& vecWeight	= GetWeight(nClass);
	vector<double>	vecNewWeight(nDataNum);

	// 重み総和
	double dSumWeight = 0.0;	
	// 弱識別器の出力
	double dScore;

	for(int nData = 0; nData < nDataNum; ++nData){
		// 分類スコアの算出
		dScore = Prediction(nRound, nClass, nData);
		// データの重みの更新
		if(vecLabel[nData] == nClass)
			vecNewWeight[nData] = vecWeight[nData] * exp(-dScore);
		else
			vecNewWeight[nData] = vecWeight[nData] * exp(dScore);
		//総和
		dSumWeight += vecNewWeight[nData];		
	}

	// 正規化
	for(int nData = 0; nData < nDataNum; ++nData)
		vecNewWeight[nData] = vecNewWeight[nData] / dSumWeight;

	m_vecWeight[nClass].swap(vecNewWeight);
}

inline double CRealAdaBoost::Prediction(int nRound, int nClass, int nData)
{
	// 弱識別器
	Classifier* lpWeakCl = GetWeakCl(nRound, nClass);
	// BINの算出
	int nBin = CompBin(lpWeakCl->HistDim, nData);
	// 分類スコア
	double dScore = lpWeakCl->Lut[nBin];
	
	return dScore;
}

inline void CRealAdaBoost::UpdateClScore(int nRound, int nClass)
{
	int nDataNum = GetDataNum();

	// 分類スコアの更新
	for(int nData = 0; nData < nDataNum; ++nData)
		m_vecScore[nClass][nData] += Prediction(nRound, nClass, nData);
}
