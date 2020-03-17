#pragma once

#include "learning.h"
#include "BaseDefine.h"

#define TREE_DEPTH		2										// 決定木の深さ
#define CLWGHT_NUM		(int)pow(2.0, TREE_DEPTH-1.0)			// 弱識別器重みの数
#define CLSSFR_NUM		((int)pow(2.0, TREE_DEPTH)-1)			// 弱識別器の数

#define SAMPLING_NUM	300

const double ThrScore_DAB[CLASS_NUM] = {0.0, -0.1, -0.3, 0.0};	// 分類スコアの閾値

using namespace std;

inline	float	round_place(float value, int dec_place=1)
{
	//小数第[dec_place]位，1の位:dec_place=-1, 10の位:dec_place=-2
	float n;
	if(dec_place > 0)
		n = float(dec_place-1);
	else
		n = float(dec_place);

	float tmp = pow(10.0f, n);
	value	*= tmp;

	float fValue = round(value);
	fValue /= tmp;

	return fValue;
}

class CAdaBoost : public CLearning
{
public:
	// 識別器(DAB)
	struct Classifier{
		vector<int>		TreeDim;		// 特徴次元
		vector<float>	ThrVal;			// 閾値
		vector<int>		IsHigh;			// High or Low
		// コンストラクタ
		Classifier(){
			TreeDim = vector<int>(CLSSFR_NUM);
			ThrVal	= vector<float>(CLSSFR_NUM);
			IsHigh	= vector<int>(CLSSFR_NUM);
		}
	};

private:
	// メンバ変数
	vector<Classifier>		m_vecWeakCl;
	vector<vector<float>>	m_vecSpltPnt;
	vector<vector<int>>		m_vecClssfrNum;			// 決定木のノード番号
	vector<vector<bool>>	m_vecClResult;			// 識別結果
	vector<vector<double>>	m_vecErrRate;			// エラー率
	vector<vector<double>>	m_vecErrWght;
	vector<vector<double>>	m_vecSumWght;
	vector<vector<double>>	m_vecClssfrWght;
	vector<vector<double>>	m_vecFtrCntrbt;

public:
	// メンバ関数
	void	Training();
	void	Test();

	// 内部使用関数
	void	Init_DAB();
	void	CompSpltPnt();
	void	SelectClassifier(int nRound, int nClass);
	void	DecisionTree(vector<vector<bool>>& vecTreeData, int nRound, int nClass);
	void	AnalyzeClssfrWght(int nRound, int nClass);
	void	DataWeightUpdate(int nRound, int nClass);
	void	UpdateClScore(int nRound, int nClass);
	void	OutCsvClssfr(int nRound);
	void	OutDatClssfr();
	void	ReadClassifier();
	void	CompTree(int nRound, int nClass);
	void	CompTrData(vector<vector<bool>>& vecTreeData, int nRound, int nClass);
	void	UpdateFtrCntrbt(int nRound, int nClass);
	void	OutCsvFtrCntrbt(int nRound);

	// 情報取得
	Classifier*				GetWeakCl(int nRound, int nClass){		return	&m_vecWeakCl[nRound*CLASS_NUM+nClass];		};
	vector<int>&			GetClssfrNum(int nClass){				return	m_vecClssfrNum[nClass];						};
	vector<bool>&			GetClResult(int nClass){				return	m_vecClResult[nClass];						};
	vector<double>&			GetErrRate(int nClass){					return	m_vecErrRate[nClass];						};
	vector<double>&			GetErrWght(int nClass){					return	m_vecErrWght[nClass];						};
	vector<double>&			GetSumWght(int nClass){					return	m_vecSumWght[nClass];						};
	vector<vector<float>>&	GetSpltPnt(){							return	m_vecSpltPnt;								};
	vector<double>&			GetClssfrWght(int nRound, int nClass){	return	m_vecClssfrWght[nRound*CLASS_NUM+nClass];	};
/*
	void ReadParameter();
	void ScoreToProb();
	void ProbTest();
	vector<vector<double>>&	GetSgmdProb(){ return m_SgmdProb; };
	vector<vector<double>>& GetTblProb(){ return m_TblProb;	};
*/
};

inline void CAdaBoost::SelectClassifier(int nRound, int nClass)
{	
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<double>& vecWeight	= GetWeight(nClass);
//	double			epsilon		= 1.0 / double(GetLblNum(nClass));
	double			epsilon		= 1.0 / double(GetDataNum());

	vector<int>		vecClssfrNum(nDataNum);
	vector<bool>	vecClResult(nDataNum, false);
	vector<double>	vecErrRate(CLWGHT_NUM);
	vector<double>	vecErrWght(CLWGHT_NUM);
	vector<double>	vecSumWght(CLWGHT_NUM);

	// 決定木による分類
	vector<vector<bool>> vecTreeData;
	DecisionTree(vecTreeData, nRound, nClass);

	double dSumWght;		//データ重み合計
	double dErrRate;		//エラー率

	for(int i = 0; i < CLWGHT_NUM; ++i){
		// 初期化
		dSumWght = 0.0;
		dErrRate = 0.0;
		for(int nData = 0; nData < nDataNum; ++nData){
			// Psitive
			if(vecTreeData[i*2+POS][nData]){
				dSumWght += vecWeight[nData];
				// 識別結果
				vecClResult[nData] = true;
				// False
				if(vecLabel[nData] != nClass)
					dErrRate += vecWeight[nData];
				// 弱識別器の番号
				vecClssfrNum[nData] = i;
			// Negative
			}else if(vecTreeData[i*2+NEG][nData]){
				dSumWght += vecWeight[nData];
				// False
				if(vecLabel[nData] == nClass)
					dErrRate += vecWeight[nData];
				// 弱識別器の番号
				vecClssfrNum[nData] = i;
			}
		}
		vecErrWght[i] = dErrRate;
		vecSumWght[i] = dSumWght;
		// 正規化
		vecErrRate[i] = dErrRate / (dSumWght+epsilon);
	}
	// 保存
	m_vecClssfrNum[nClass].swap(vecClssfrNum);
	m_vecClResult[nClass].swap(vecClResult);
	m_vecErrRate[nClass].swap(vecErrRate);
	m_vecErrWght[nClass].swap(vecErrWght);
	m_vecSumWght[nClass].swap(vecSumWght);
}

inline void CAdaBoost::DecisionTree(vector<vector<bool>>& vecTreeData, int nRound, int nClass)
{
	int						nDataNum	= GetDataNum();
	int						nDimNum		= GetDimNum();
	vector<int>&			vecLabel	= GetLabel();
	vector<vector<float>>&	vecSpltPnt	= GetSpltPnt();
	vector<vector<float>>&	vecFeature	= GetFeature();
	vector<vector<double>>&	vecWeight	= GetWeight();

	// ノード数:1，全要素:trueで初期化
	vector<vector<bool>> vecOldTrData(1, vector<bool>(nDataNum, true));
	int	Count = 0;

	int		nTreeDim;
	float	fThrVal;
	int		nTrIsHigh;
	double	dErrRate, dErrRateMin;
	int		nSpPntSize, nSpltNum;

	// 決定木
	for(int nDepth = 0; nDepth < TREE_DEPTH; ++nDepth){
		vector<vector<bool>> vecNewTrData(pow(2.0, nDepth+1), vector<bool>(nDataNum, false));
		for(int i = 0; i < vecOldTrData.size(); ++i){
			dErrRateMin = DBL_MAX;
			for(int nDim = 0; nDim < nDimNum; ++nDim){
				// 分割候補数
				nSpPntSize = vecSpltPnt[nDim].size();

				for(int nSample = 0; nSample < SAMPLING_NUM; ++nSample){

					// 候補数を超えたら終了
					if(nSample > nSpPntSize-1)
						break;

					// 指定数より多い場合はランダム
					if(nSpPntSize > SAMPLING_NUM)
						nSpltNum = (int)(rand() / (RAND_MAX+1.0)*nSpPntSize);
					else
						nSpltNum = nSample;

					// High or Low
					for(int nIsHigh=0; nIsHigh<2; ++nIsHigh){

						dErrRate = 0.0;
						for(int nData = 0; nData < nDataNum; ++nData){
							if(!vecOldTrData[i][nData])
								continue;
							// trueの場合
							if(nIsHigh == (vecFeature[nDim][nData]>vecSpltPnt[nDim][nSpltNum])){
								// Negative
								if(vecLabel[nData] != nClass)
									dErrRate += vecWeight[nClass][nData];
							// falseの場合
							}else{
								// Positive
								if(vecLabel[nData] == nClass)
									dErrRate += vecWeight[nClass][nData];
							}
						}
						if(dErrRate < dErrRateMin){
							nTreeDim	= nDim;
							nTrIsHigh	= nIsHigh;
							fThrVal		= vecSpltPnt[nDim][nSpltNum];
							dErrRateMin	= dErrRate;
						}
					}
				}
			}
			// 分類
			for(int nData = 0; nData < nDataNum; ++nData){
				if(!vecOldTrData[i][nData])
					continue;
				if(nTrIsHigh == (vecFeature[nTreeDim][nData] > fThrVal))
					vecNewTrData[i*2+POS][nData] = true;
				else
					vecNewTrData[i*2+NEG][nData] = true;
			}
			// 保存
			m_vecWeakCl[nRound*CLASS_NUM + nClass].TreeDim[Count]	= nTreeDim;
			m_vecWeakCl[nRound*CLASS_NUM + nClass].IsHigh[Count]	= nTrIsHigh;
			m_vecWeakCl[nRound*CLASS_NUM + nClass].ThrVal[Count]	= fThrVal;
			++Count;
		}
		vecOldTrData.swap(vecNewTrData);
	}
	vecTreeData.swap(vecOldTrData);
}

inline void CAdaBoost::AnalyzeClssfrWght(int nRound, int nClass)
{
	vector<double>& vecErrRate	= GetErrRate(nClass);
	vector<double>& vecErrWght	= GetErrWght(nClass);
	vector<double>& vecSumWght	= GetSumWght(nClass);
//	double			epsilon		= 1.0 / double(GetLblNum(nClass));
	double			epsilon		= 1.0 / double(GetDataNum());

	vector<double> vecClssfrWght(CLWGHT_NUM);
	
	for(int i = 0; i < CLWGHT_NUM; ++i)
		vecClssfrWght[i] = 0.5 * log((vecSumWght[i]-vecErrWght[i]+epsilon) / (vecErrWght[i]+epsilon));
//		vecClssfrWght[i] = 0.5 * log((1.0-vecErrRate[i]+epsilon) / (vecErrRate[i]+epsilon));

	// 保存
	m_vecClssfrWght[nRound*CLASS_NUM + nClass].swap(vecClssfrWght);
}

inline void CAdaBoost::DataWeightUpdate(int nRound, int nClass)
{
	int				nDataNum		= GetDataNum();
	vector<int>&	vecLabel		= GetLabel();
	vector<int>&	vecClssfrNum	= GetClssfrNum(nClass);
	vector<bool>&	vecClResult		= GetClResult(nClass);
	vector<double>&	vecWeight		= GetWeight(nClass);
	vector<double>& vecClssfrWght	= GetClssfrWght(nRound, nClass);

	int		nClssfrNum;
	double	tmp;

	for (int nData = 0; nData < nDataNum; ++nData){
		tmp = -1.0;
		// 決定機のノード番号
		nClssfrNum = vecClssfrNum[nData];
		// Psitive
		if(vecClResult[nData]){
			// flase
			if(vecLabel[nData] != nClass)
				tmp = 1.0;
		// Negative
		}else{ 
			// flase
			if(vecLabel[nData] == nClass)
				tmp = 1.0;
		}
		m_vecWeight[nClass][nData] = vecWeight[nData] * exp(vecClssfrWght[nClssfrNum]*tmp);
	}
}

inline void CAdaBoost::UpdateClScore(int nRound, int nClass)
{
	int				nDataNum		= GetDataNum();
	vector<int>&	vecClssfrNum	= GetClssfrNum(nClass);
	vector<double>& vecClssfrWght	= GetClssfrWght(nRound, nClass);
	vector<bool>&	vecClResult		= GetClResult(nClass);

	int		nClssfrNum;
	double	tmp;

	for(int nData = 0; nData < nDataNum; ++nData){
		nClssfrNum	= vecClssfrNum[nData];
		tmp			= vecClResult[nData] ? 1.0 : -1.0;
		// 更新
		m_vecScore[nClass][nData] += vecClssfrWght[nClssfrNum] * tmp;
	}
}

inline void CAdaBoost::UpdateFtrCntrbt(int nRound, int nClass)
{
	// 情報取得
	vector<double>& vecClssfrWght	= GetClssfrWght(nRound, nClass);
	Classifier*		lpWeakCl		= GetWeakCl(nRound, nClass);

	// 特徴次元
	int nDim = lpWeakCl->TreeDim[0];
	// 貢献度
	m_vecFtrCntrbt[nClass][nDim] += vecClssfrWght[0];
}