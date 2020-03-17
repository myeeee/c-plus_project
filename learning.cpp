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
	
	// �e�N���X�̃��x����
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

	// �G���[���o�̓t�@�C���̏�����
	string strFileName = GetFoldName() + "\\ErrorRate - MaxScoreClass.csv";
	InitErrFile(strFileName);
	// �G���[���o�̓t�@�C���̏�����
	strFileName = GetFoldName() + "\\ErrorRate - ThrPositiveClass.csv";
	InitErrFile(strFileName);

	m_vecScore = vector<vector<double>>(CLASS_NUM, vector<double>(nDataNum, 0.0));

	//�ۑ�
	m_nDataNum	= nDataNum;
	m_nDimNum	= nDimNum;
	m_vecFeature.swap(vecFeature);
	m_vecLabel.swap(vecLabel);
	m_vecLblNum.swap(vecLblNum);
}

void CLearning::WeightIniOld()
{
	int nDataNum = GetDataNum();

	// (�f�[�^�d��) = 1.0 / (�S�f�[�^��)
	double dWeight	= 1.0 / (double)nDataNum;
	m_vecWeight = vector<vector<double>>(CLASS_NUM, vector<double>(nDataNum, dWeight));
}

void CLearning::WeightIni()
{
	int				nDataNum	= GetDataNum();
	vector<int>&	vecLabel	= GetLabel();
	vector<int>&	vecLblNum	= GetLblNum();

	// (�e�N���X�̃f�[�^�d��) = 1.0/(�e�N���X�f�[�^��)
	vector<double> vecDtWeight(CLASS_NUM);
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		vecDtWeight[nClass] = 1.0/(double)vecLblNum[nClass];
	
	// �f�[�^�d��
	vector<double> vecWeight(nDataNum);
	int tmp;
	for(int nData = 0; nData < nDataNum; ++nData){
		tmp = vecLabel[nData];
		vecWeight[nData] = vecDtWeight[tmp];
	}

	// �R�s�[
	m_vecWeight = vector<vector<double>>(CLASS_NUM);
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		m_vecWeight[nClass] = vecWeight;
}

void CLearning::UpdateClResult()
{
	// ���擾
	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>& vecScore = GetScore();

	vector<int> vecMxScrClss(nDataNum);				// ���ʃN���X
	vector<int>	vecMxScrClNum(CLASS_NUM, 0);		// ���o��
	vector<int>	vecMxScrTpNum(CLASS_NUM, 0);		// TruPositive��

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
		// �ő�X�R�A�̃N���X�Ɏ���
		vecMxScrClss[nData] = tmp;
		if(tmp != -1){
			// ���o��
			++vecMxScrClNum[tmp];
			//TruePositive
			if(tmp == vecLabel[nData])
				++vecMxScrTpNum[tmp];
		}
	}

	// �ۑ�
	m_vecMxScrClss.swap(vecMxScrClss);
	m_vecMxScrClNum.swap(vecMxScrClNum);
	m_vecMxScrTpNum.swap(vecMxScrTpNum);
}

void CLearning::UpdateClResult(const double* lpThrScore)
{
	// ���擾
	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>& vecScore = GetScore();

	vector<vector<bool>>	vecThrPos(CLASS_NUM, vector<bool>(nDataNum, false));
	vector<int>				vecThrPsNum(CLASS_NUM, 0);		// �e�N���X���o��
	vector<int>				vecThrPsTpNum(CLASS_NUM, 0);	// �e�N���XTruPositive��

	for(int nData = 0; nData < nDataNum; ++nData){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// ���ރX�R�A��臒l�ȏ�ł���Ό��o
			if(vecScore[nClass][nData] > lpThrScore[nClass]){
				vecThrPos[nClass][nData] = true;
				// ���o��
				++vecThrPsNum[nClass];
				// TruePositive
				if(vecLabel[nData] == nClass)
					++vecThrPsTpNum[nClass];
			}
		}
	}

	// �ۑ�
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
		double dRcll;	//�Č���
		double dPrcsn;	//�K����
		fsOutCsv << nRound << ",";
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			dRcll	= (double)vecTrPsNum[nClass] / (double)vecLblNum[nClass];
			dPrcsn	= (double)vecTrPsNum[nClass] / (double)vecClssNum[nClass];
			//�o��
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
		//�N���X���Ƃɏo��
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			fsOutCsv << "Class" << nClass << endl;
			//�����ʂ��o��
			for(int nDim = 0; nDim < nDimNum; ++nDim){
				fsOutCsv << "DIM" << nDim << endl;
				for(int nData = 0 ; nData < nDataNum; ++nData){
					if(vecLabel[nData] == nClass)
						fsOutCsv << vecFeature[nDim][nData] << ",";
				}
				fsOutCsv << endl;
			}
			fsOutCsv << endl;
			//���ރX�R�A���o��
			for(int i = 0; i < CLASS_NUM; ++i){
				fsOutCsv << "Score" << i <<  endl;
				for(int nData = 0 ; nData < nDataNum; ++nData){
					if(vecLabel[nData] == nClass)
						fsOutCsv << vecScore[i][nData] << ",";
				}
				fsOutCsv << endl;
			}
			fsOutCsv << endl;
			//���ʌ��ʂ��o��
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
		//���ރX�R�A�̍ő�E�ŏ��l
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
		//�e�X�R�A臒l�ɂ�����Recall��Precision
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
						//���o��
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
					//�o��
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
	//�N���X�m���\�̍쐬
	CompProbTbl();
	//���W�X�e�B�b�N��A����
	LogRegression();
}

void CLearning::CompProbTbl()
{
	m_vecScrTbl	= vector<vector<double>>(CLASS_NUM);
	m_vecPrbTbl	= vector<vector<double>>(CLASS_NUM);

	int						nDataNum = GetDataNum();
	vector<int>&			vecLabel = GetLabel();
	vector<vector<double>>&	vecScore = GetScore();

	//���ރX�R�A�������ɕϊ�
	vector<vector<double>> vecSortScr = m_vecScore;
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass)
		sort(vecSortScr[nClass].begin(), vecSortScr[nClass].end());

	int nSpltPnt = nDataNum / SCORE_BIN;

	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		// ������
		vector<double>	vecScrBin(SCORE_BIN);
		vector<double>	vecPrbTbl(SCORE_BIN);
		vector<int>		vecCount(SCORE_BIN, 0);
		int nSpltNum = 0;
		// BIN�̍쐬
		for(int i = 0; i < SCORE_BIN; ++i){
			nSpltNum += nSpltPnt;
			if(nSpltNum > nDataNum-1)
				nSpltNum = nDataNum - 1;
			vecScrBin[i] = vecSortScr[nClass][nSpltNum];
		}
		// �eBIN�̃f�[�^�����J�E���g
		size_t stBin;
		vector<double>::iterator itr;
		for(int nData = 0; nData < nDataNum; ++nData){
			if(vecLabel[nData] != nClass)
				continue;
			// �񕪒T��
			itr		= upper_bound(vecScrBin.begin(), vecScrBin.end(), vecScore[nClass][nData]);
			stBin	= distance(vecScrBin.begin(), itr);
			if(stBin > SCORE_BIN-1)
				stBin = SCORE_BIN-1;
			++vecCount[stBin];
		}
		// �N���X�m���\
		for(int i = 0; i < SCORE_BIN; ++i)
			vecPrbTbl[i] = (double)vecCount[i]/(double)nSpltPnt;
		// �ۑ�
		m_vecScrTbl[nClass].swap(vecScrBin);
		m_vecPrbTbl[nClass].swap(vecPrbTbl);
	}

	// csv�t�@�C���ɏo��
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\ProbTable.csv", ios::out);
	if (fsOutCsv.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++ nClass){
			fsOutCsv << "Class" << nClass << endl;
			fsOutCsv << "Score" << endl;
			// �o��
			for(int i = 0; i < SCORE_BIN; ++i)
				fsOutCsv << m_vecScrTbl[nClass][i] << ",";
			fsOutCsv << endl;
			// �o��
			fsOutCsv << "Probability" << endl;
			for(int i = 0; i < SCORE_BIN; ++i)
				fsOutCsv << m_vecPrbTbl[nClass][i] << ",";
			fsOutCsv << endl;
		}
		fsOutCsv.close();
	}

	//dat�t�@�C���ɏo��
	fstream fsOutDat;
	fsOutDat.open(GetFoldName() + "\\ProbTable.dat", ios::out | ios::binary);
	if (fsOutDat.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			//�o��
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

	//�i���j
	vector<double>	pi(nDataNum);
	double			a, Lw, DrvLw, Gradw0, Gradw1;
	double eta		= 0.00001;
	double epsilon	= 0.00000001;

	// �v�Z&csv�t�@�C���ɏo��
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\LogisticRegression.csv", ios::out);
	if (fsOutCsv.is_open()){

		for(int nClass = 0; nClass < CLASS_NUM; ++ nClass){
			cout << "\nClass" << nClass << endl;
			fsOutCsv << "\nClass" << nClass << "\nLw, DrvLw, w0, w1" << endl;

			// ���t�f�[�^�̍쐬
			vector<int> t(nDataNum, 0);
			for(int nData = 0; nData < nDataNum; ++nData){
				if(vecLabel[nData] == nClass){
					t[nData] = 1;
				}
			}
			//�i���j
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

			// �ۑ�
			m_vecSgmdPrm[nClass][0] = w0;
			m_vecSgmdPrm[nClass][1] = w1;
		}
		fsOutCsv.close();
	}

	//dat�t�@�C���ɏo��
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
	// �V�O���C�h�֐��̃p�����[�^
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

	//�V�O���C�h�֐����m���ɕϊ�
	double dAlpha;
	for(int nData = 0; nData < nDataNum; ++nData){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			dAlpha = exp(vecSgmdParam[nClass][0] + vecScore[nClass][nData]*vecSgmdParam[nClass][1]);
			vecProb[nClass][nData] = dAlpha / (1.0 + dAlpha);
		}
	}

	vecClProb.swap(vecProb);

	// ���o�^
//	m_vecScore = vecClProb;
//	// �ő啪�ރX�R�A�ɂ�鎯�ʌ���
//	UpdateClResult();
//	// ���ʌ��ʂ��o��
//	int nRound = LEARN_NUM-1;
//	string strFileName = "ErrorRate - ClassProbabilityClassify.csv";
//	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);
}

void CLearning::TimeSeriesClassify(vector<vector<double>>& vecLogOdds)
{
	// ���o�^
	m_vecScore.swap(vecLogOdds);

	// �ő啪�ރX�R�A�ɂ�鎯�ʌ���
	UpdateClResult();

	// ���ʌ��ʂ��o��
	int nRound = LEARN_NUM-1;
	string strFileName = GetFoldName() + "\\ErrorRate - TimeSeriesClassify.csv";
	OutputErr(strFileName, m_vecMxScrClNum, m_vecMxScrTpNum, nRound);

	// ���ʂ̏ڍׂ��o��
	OutCsvDetError();

	// Recall Precision Curve
	DrawCurve();
}

void CLearning::AnalyzeRcPrCurve(vector<int>& vecLimitLbl, vector<vector<double>>& vecLimitScr)
{
	int nDataNum = vecLimitLbl.size();

	// �e�N���X�̃��x����
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

	// �ő啪�ރX�R�A�ɂ�鎯�ʌ���
	UpdateClResult();

	// ���ʌ��ʂ��o��
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
	// ���ݎ���
	time_t	t	= time(NULL);
    tm*		ptr	= localtime(&t);
	stringstream ssTime;
	ssTime << ptr->tm_hour << ":" << ptr->tm_min << ":" << ptr->tm_sec;

	return ssTime.str();
}