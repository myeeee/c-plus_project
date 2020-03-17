#pragma once

#include "RealAdaBoost.h"

using namespace std;

void CRealAdaBoost::CompSpltPnt()
{
	int nDimNum		= GetDimNum();
	int nDataNum	= GetDataNum();
	
	vector<vector<float>> vecSpltPnt(nDimNum, vector<float>(BIN[0]+BIN[1]));

	// �R�s�[
	vector<vector<float>> vecSortFtr = m_vecFeature;
	// �����ɕϊ�
	for(int nDim = 0; nDim < nDimNum; ++nDim)
		sort(vecSortFtr[nDim].begin(), vecSortFtr[nDim].end());

	// (BIN�̕�) = (���f�[�^��)
	for(int i = 0; i < HIST_DIM; ++i){
		//���f�[�^��
		int nSpltDtNum = int(nDataNum / BIN[i]);

		for(int nDim = 0; nDim < nDimNum; ++nDim){	
			for(int nBin = 0; nBin < BIN[i]; ++nBin)
				vecSpltPnt[nDim][i*BIN[0] + nBin] = vecSortFtr[nDim][(nBin+1)*nSpltDtNum];
		}
	}
	// �ۑ�
	m_vecSpltPnt.swap(vecSpltPnt);

	// BIN��csv�t�@�C���ɏo��
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
	// �������m��
	m_vecWeakCl		= vector<Classifier>(LEARN_NUM * CLASS_NUM); 

	// ���ʊ�o�̓t�@�C��(csv)��������
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
			// BIN���ꗗ�̃C�e���[�^
			itrStrt = vecSpltPnt[nDim].begin() + i*BIN[0];
			// �������x�����߂�
			for(int nData = 0; nData < nDataNum; ++nData){
				// �񕪒T��
				itrBound = upper_bound(itrStrt, itrStrt+BIN[i], vecFeature[nDim][nData]);
				// BIN�̔ԍ�
				stBin = distance(itrStrt, itrBound);
				if(stBin > BIN[i] - 1)
					stBin = BIN[i] - 1;

				vecBinNum[i*nDimNum + nDim][nData] = stBin;
			}
		}
	}
	// �ۑ�
	m_vecBinNum.swap(vecBinNum);
}

void CRealAdaBoost::Training()
{
	m_strFoldName = "Training_RAB";
	_mkdir(m_strFoldName.c_str());

	// ������
	Init_RAB();
	m_vecHist		= vector<vector<vector<double>>>(CLASS_NUM);
	m_vecHistDtNum	= vector<vector<vector<int>>>(CLASS_NUM);
	// �f�[�^�d�݂̏�����
	WeightIniOld();
	//WeightIni();
	// �m�����x���z��BIN�����쐬
	CompSpltPnt();
	// �������x�����߂�
	CreateFtrStrngth();
	// �w�K
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;
		
		#pragma omp parallel for
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// �㎯�ʊ�̑I��
			SelectClassifier(nRound, nClass);
			// �f�[�^�d�݂̍X�V
			DataWeightUpdate(nRound, nClass);
			// ���ރX�R�A�̍X�V
			UpdateClScore(nRound, nClass);
		}
		// �ő啪�ރX�R�A�ɂ�鎯�ʌ���
		UpdateClResult();
		// 臒l�ɂ�鎯�ʌ���
		UpdateClResult(ThrScore_RAB);
		// csv�t�@�C���ɃG���[�����o��
		OutCsvError(nRound);
		// csv�t�@�C���Ɏ��ʊ���o��
		OutCsvClssfr(nRound);
		// �m�����x���z���o��
		OutCsvHist(nRound);
	}
	// ���ʊ��dat�t�@�C���ɏo��
	OutDatClssfr();
	// ���ʂ̏ڍׂ��o��
	OutCsvDetError();
	// �����ʂ��o��
	OutCsvFeature();
	// Recall Precision Curve
	DrawCurve();
}

void CRealAdaBoost::Test()
{
	m_strFoldName = "Test_RAB";
	_mkdir(m_strFoldName.c_str());

	// ������
	Init_RAB();
	// ���ʊ�̓ǂݍ���
	ReadClassifier();
	// �������x�����߂�
	CreateFtrStrngth();
	// Test
	for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
		cout << " " << nRound;
		
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			// ���ރX�R�A�̍X�V
			UpdateClScore(nRound, nClass);
		}
		// �ő啪�ރX�R�A�ɂ�鎯�ʌ���
		UpdateClResult();
		// 臒l�ɂ�鎯�ʌ���
		UpdateClResult(ThrScore_RAB);
		// csv�t�@�C���ɃG���[�����o��
		OutCsvError(nRound);
		// csv�t�@�C���Ɏ��ʊ���o��
		OutCsvClssfrOld(nRound);
	}
	// ���ʊ��dat�t�@�C���ɏo��
	OutDatClssfr();
	// ���ʂ̏ڍׂ��o��
	OutCsvDetError();
	// �����ʂ��o��
	OutCsvFeature();
	// Recall Precision Curve
	DrawCurve();
	// ���ނɂ���������ʂ̊�^�����v�Z
	AnalyzeFtrScr();
}

void CRealAdaBoost::ReadClassifier()
{
	int nDimNum = GetDimNum();

	vector<vector<float>> vecSpltPnt(nDimNum, vector<float>(BIN[0]+BIN[1]));

	//�㎯�ʊ�̓ǂݍ���
	fstream fsReadCl;
	fsReadCl.open("Training_RAB\\Classifier_RAB.dat", ios::in | ios::binary);
	if (fsReadCl.is_open()){
		vector<int>		HistDim(HIST_DIM);
		vector<double>	Lut(PREDICT_NUM);
		
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				//�ǂݍ���
				fsReadCl.read((char*)&HistDim[0], sizeof(int)*HIST_DIM);
				fsReadCl.read((char*)&Lut[0], sizeof(double)*PREDICT_NUM);
				//�ۑ�
				m_vecWeakCl[nRound*CLASS_NUM + nClass].HistDim	= HistDim;
				m_vecWeakCl[nRound*CLASS_NUM + nClass].Lut		= Lut;
			}
		}
		fsReadCl.close();
	}

	//BIN�̓ǂݍ���
	fstream fsReadBin;
	fsReadBin.open("Training_RAB\\Bin_RAB.dat", ios::in | ios::binary);
	if (fsReadBin.is_open()){
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsReadBin.read((char*)&vecSpltPnt[nDim][0], sizeof(float)*(BIN[0]+BIN[1]));
		fsReadBin.close();
	}
	// �ۑ�
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
			// ���擾
			lpWeakCl = GetWeakCl(nRound, nClass);
			int i;
			// ��������
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

				//�o��
				fsOutDatCl.write((const char*)&HistDim[0], sizeof(int)*HIST_DIM);
				fsOutDatCl.write((const char*)&Lut[0], sizeof(double)*PREDICT_NUM);
			}
		}
		fsOutDatCl.close();
	}

	//BIN���o��
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
		// ������
		dSumScr = 0.0;
		// ���ʃN���X
		tmp = vecMxScrClss[nData];
		if(tmp == -1)
			continue;
		// ���ʃX�R�A��ݐ�
		for(int nRound = 0; nRound < LEARN_NUM; ++nRound){
			// �X�R�A
			dScore = Prediction(nRound, tmp, nData);
			// �㎯�ʊ�
			lpWeakCl = GetWeakCl(nRound, tmp);
			for(int i = 0; i < HIST_DIM; ++i){
				// ��������
				nHistDim = lpWeakCl->HistDim[i];
				// �ݐ�
				vecFtrScr[nData][nHistDim] += dScore;
				// ���X�R�A
				dSumScr += dScore;
			}
		}
		// �ŏ��l
		dMinScr = DBL_MAX;
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			if(vecFtrScr[nData][nDim] < dMinScr)
				dMinScr = vecFtrScr[nData][nDim];
		}

		// �X�R�A - �ŏ��l
		dSumNrmScr = 0.0;
		for(int nDim = 0; nDim < nDimNum; ++nDim){
			vecNrmFtrScr[nData][nDim] = vecFtrScr[nData][nDim] - dMinScr;
			dSumNrmScr += vecNrmFtrScr[nData][nDim];
		}
		if(dSumNrmScr == 0.0)
			continue;
		// ���K��
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			vecNrmFtrScr[nData][nDim] /= dSumNrmScr;

		if(dSumScr == 0.0)
			continue;
		// ���K��
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			vecFtrScr[nData][nDim] /= dSumScr;
	}

	// Csv�ɏo��
	fstream fsOutCsvNrm;
	fsOutCsvNrm.open(GetFoldName() + "\\NrmFeatureScore.csv", ios::out);
	if (fsOutCsvNrm.is_open()){
		// �����������o��
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsOutCsvNrm << nDim << ",";
		fsOutCsvNrm << endl;
		// ���x��
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			// ���ʃN���X
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){				
				// nLabel->nClass�ɕ��ނ��ꂽ�f�[�^
				for(int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData] != nLabel || vecMxScrClss[nData] != nClass)
						continue;
					// �o��
					fsOutCsvNrm << nLabel << " -> " << nClass << endl;
					for(int nDim = 0; nDim < nDimNum; ++nDim)
						fsOutCsvNrm << vecNrmFtrScr[nData][nDim] << ",";
					fsOutCsvNrm << endl;
				}
			}
		}
		fsOutCsvNrm.close();
	}

	// Csv�ɏo��
	fstream fsOutCsv;
	fsOutCsv.open(GetFoldName() + "\\FeatureScore.csv", ios::out);
	if (fsOutCsv.is_open()){
		// �����������o��
		for(int nDim = 0; nDim < nDimNum; ++nDim)
			fsOutCsv << nDim << ",";
		fsOutCsv << endl;
		// ���x��
		for(int nLabel = 0; nLabel < CLASS_NUM; ++nLabel){
			// ���ʃN���X
			for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
				// ������
				vector<double> vecSumFtrScr(nDimNum, 0.0);
				// nLabel->nClass�ɕ��ނ��ꂽ�f�[�^
				for(int nData = 0; nData < nDataNum; ++nData){
					if(vecLabel[nData] != nLabel || vecMxScrClss[nData] != nClass)
						continue;
					for(int nDim = 0; nDim < nDimNum; ++nDim)
						vecSumFtrScr[nDim] += vecNrmFtrScr[nData][nDim];
						//vecSumFtrScr[nDim] += vecFtrScr[nData][nDim];
				}
				// �o��
				fsOutCsv << nLabel << " -> " << nClass << endl;
				for(int nDim = 0; nDim < nDimNum; ++nDim)
					fsOutCsv << vecSumFtrScr[nDim] << ",";
				fsOutCsv << endl;
			}
		}
		fsOutCsv.close();
	}

	// �ۑ�
	//m_vecFtrScr.swap(vecFtrScr);
	m_vecFtrScr.swap(vecNrmFtrScr);
}

/*
void CRealAdaBoost::ReadParameter()
{
	//�V�O���C�h�֐��̃p�����[�^��dat�t�@�C���ɏo��
	fstream ReadSigmoid;
	ReadSigmoid.open("Sigmoid Param.dat", ios::in | ios::binary);
	if (ReadSigmoid.is_open()){
		for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
			ReadSigmoid.read((char*)&m_SgmdParam[nClass][0], sizeof(double)*2);
		}
		ReadSigmoid.close();
	}

	//����m���\��dat�t�@�C���ɏo��
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

	//�V�O���C�h
	for(int nClass = 0; nClass < CLASS_NUM; ++nClass){
		for(int nData = 0; nData < m_DataNum; ++nData){
			m_SgmdProb[nClass][nData] = exp(m_SgmdParam[nClass][0] + m_vecH[nClass][nData]*m_SgmdParam[nClass][1]) / (1 + exp(m_SgmdParam[nClass][0] + m_vecH[nClass][nData]*m_SgmdParam[nClass][1]));
		}
	}

	//����m���\
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
	double MaxH; //�ő�X�R�A

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

		//�N���X����
		m_vecClass[nData] = tmp;
		if(tmp != -1){
			//++���o��
			++m_ClssNum[tmp];
			//++TruePositive
			if(tmp == m_vecLabel[nData]){
				++m_TrPsNum[tmp];
			}
		}
	}

	//�G���[�ڍׂ������o��
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