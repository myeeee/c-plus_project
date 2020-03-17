#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <time.h>
#include <sstream>
#include <direct.h>

#define		LEARN_NUM		350		// 学習回数
#define		SCORE_BIN		32
#define		POS				1
#define		NEG				0

using namespace std;

// クラス名
enum CLASS
{
	CAR, PEDESTRIAN, BICYCLIST, BACKGROUND, CLASS_NUM
};

class CLearning
{
protected:
	// メンバ変数
	int						m_nDataNum;
	int						m_nDimNum;
	vector<vector<float>>	m_vecFeature;
	vector<int>				m_vecLabel;
	vector<vector<double>>	m_vecWeight;
	vector<vector<double>>	m_vecScore;
	vector<int>				m_vecLblNum;			// ラベル数
	vector<int>				m_vecMxScrClss;			// 最大スコアにより検出したクラス
	vector<int>				m_vecMxScrClNum;		// 最大スコアによる検出数
	vector<int>				m_vecMxScrTpNum;		// 最大スコアによるTruePositive数
	vector<vector<bool>>	m_vecThrPos;			// 閾値による検出(true or false)
	vector<int>				m_vecThrPsNum;			// 閾値による検出数
	vector<int>				m_vecThrPsTpNum;		// 閾値によるTruePositive数
	vector<vector<double>>	m_vecScrTbl;
	vector<vector<double>>	m_vecPrbTbl;
	vector<vector<double>>	m_vecSgmdPrm;

	string					m_strFoldName;

public:
	// メンバ関数
	void	Initialize(vector<vector<float>>& vecFeature, vector<int>& vecLabel);
	void	InitErrFile(string& strFileName);
	void	WeightIniOld();
	void	WeightIni();
	void	UpdateClResult();
	void	UpdateClResult(const double* lpThrScore);
	void	OutCsvError(int nRound);
	void	OutputErr(string& strFileName, vector<int>& vecClssNum, vector<int>& vecTrPsNum, int nRound);
	void	OutCsvDetError();
	void	OutCsvFeature();
	void	DrawCurve();
	void	AnalyzeClProb();
	void	CompProbTbl();
	void	LogRegression();

	void	ReadSgmdParam();
	void	Score2Prob(vector<vector<double>>& vecScore, vector<vector<double>>& vecClProb);
	void	TimeSeriesClassify(vector<vector<double>>& vecLogOdds);
	void	WriteClResult(string strFileName, vector<int>& vecClass);
	void	ReadClResult(string strFileName, vector<int>& vecClass);
	void	AnalyzeRcPrCurve(vector<int>& vecLimitLbl, vector<vector<double>>& vecLimitScr);

	// 情報取得
	int				GetDataNum(){				return	m_nDataNum;				};
	int				GetDimNum(){				return	m_nDimNum;				};
	vector<int>&	GetLabel(){					return	m_vecLabel;				};
	vector<int>&	GetLblNum(){				return	m_vecLblNum;			};
	int				GetLblNum(int nClass){		return	m_vecLblNum[nClass];	};
	vector<int>&	GetMxScrClss(){				return	m_vecMxScrClss;			};
	vector<vector<double>>& GetWeight(){			return	m_vecWeight;			};
	vector<double>&			GetWeight(int nClass){	return	m_vecWeight[nClass];	};
	vector<vector<double>>& GetScore(){				return	m_vecScore;				};
	vector<double>&			GetScore(int nClass){	return	m_vecScore[nClass];		};
	vector<vector<bool>>&	GetThrPos(){			return	m_vecThrPos;			};
	vector<vector<float>>&	GetFeature(){			return	m_vecFeature;			};
	
	vector<vector<double>>& GetSgmdParam(){			return	m_vecSgmdPrm;			};
	string&					GetFoldName(){			return	m_strFoldName;			};

	static string GetCrrntTime();
};