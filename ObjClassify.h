//////////////////////////////////////////////////////////////////////////////////////////////////
//		ObjClassify.h�ֈڍs
//////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include	<fstream>
#include	"ObjTrck.h"

#define CLASS_NUM		4		//�N���X��
#define LEARN_NUM_TRCK	73		//�w�K�� - �g���b�N����	// ��	JARI�p�̐ݒ�58

#define PROB_DIM		2		//�m�����x���z�̎�����

const int BIN[PROB_DIM]	= {64, 20};							// ��
const int PREDICT_NUM	= BIN[0]*BIN[1];

#define PIXEL_SIZE		0.06f

#define COVMTRX_ELNUM	6		//3���������U�s��v�f��
#define INRMTRX_ELNUM	6		//3�����������[�����g�s��v�f��

#define PNTHST_PTCH		0.6f	
#define PNTHST_LNGRNG	6.6f
#define PNTHST_NRRWRNG	5.4f

#define PNTHST_LNGDIM	int(0.5 + PNTHST_LNGRNG/PNTHST_PTCH)
#define PNTHST_NRRWDIM	int(0.5 + PNTHST_NRRWRNG/PNTHST_PTCH)

#define INTHST_PTCH		16.0f
#define INTHST_DIM		int(0.5 + 256.0f/INTHST_PTCH)

#define SLICE_PITCH		0.2f
#define SLICE_MINHGHT	-0.1f
#define SLICE_MAXHGHT	2.3f
#define LAYERNUM		int(0.5 + (SLICE_MAXHGHT - SLICE_MINHGHT) / SLICE_PITCH)

// �����ʃ��X�g
enum Feature
{
	//�傫��
	LONGSIDE,
	NARROWSIDE,
	HEIGHT,
	//�|�C���g��
	POINTNUM,
	//���ˋ��x���ρE���U�l
	INTENSITY,
	VARINTENSITY,
	//�ʕ�́i���ʁj
	SIDEAREA,
	SIDEPER,
	//�ʕ�́i���ʁj
	FRONTAREA,
	FRONTPER,
	//�ʕ�́i��ʁj
	TOPAREA,
	TOPPER,
	//2�l�摜�ʐ�
	SD_NONZERO,
	FRNT_NONZERO,
	TP_NONZERO,
	//�|�����킹
	DIV_VARINT,
	PNTDENSITY,
	VOLBOX,
	MUL_LNGNRRW,
	MUL_LNGHGHT,
	DIV_HGHTLNG,
	DIV_HGHTNRRW,

	SDDENS,
	FRNTDENS,
	TPDENS,

	SDCIRCLE,
	FRNTCIRCLE,
	TPCIRCLE,

	SDELONG,
	FRNTELONG,
	TPELONG,

	SDDENS_NONZ,
	FRNTDENS_NONZ,
	TPDENS_NONZ,

	SDELONG_NONZ,
	FRNTELONG_NONZ,
	TPELONG_NONZ,
	//3���������U�s��
	COVMTRXSTRT,
	COVMTRXEND = COVMTRXSTRT + COVMTRX_ELNUM - 1,
	//3�����������[�����g�s��
	INRMTRXSTRT,
	INRMTRXEND = INRMTRXSTRT + INRMTRX_ELNUM - 1,
	//�|�C���g�N���E�h���z
	PNTHSTLNGSTRT,
	PNTHSTLNGEND = PNTHSTLNGSTRT + PNTHST_LNGDIM - 1,
	PNTHSTNRRWSTRT,
	PNTHSTNRRWEND = PNTHSTNRRWSTRT + PNTHST_NRRWDIM - 1,
	//���ˋ��x���z
	INTHSTSTRT,
	INTHSTEND = INTHSTSTRT + INTHST_DIM - 1,
	//�ő�E�ŏ����ˋ��x
	INTMAX,
	INTMIN,
	//�X���C�X������
	SLICE_LWSTRT,
	SLICE_LWEND = SLICE_LWSTRT + LAYERNUM - 1,
	SLICE_NWSTRT,
	SLICE_NWEND = SLICE_NWSTRT + LAYERNUM - 1,
	SLICE_AISTRT,
	SLICE_AIEND = SLICE_AISTRT + LAYERNUM - 1,
	SLICE_PAI,
	SLICE_PVI,
	//���x
	VELOCITY,
	//���x�̕��U
	VELVAR,

	// �ʐύŏ���`
	BOXAREA_TP,							// ��
	// Box�̐� 
	MINBOXVOL,							// ��

	DIM_NUM,
	
	//�n�ʂ���̍ő�E�ŏ�����
	MAXHEIGHT,
	MINHEIGHT,
	//���Ԃ���̋���
	DISTANCE,

	ALLFTR_NUM
};


//���ʊ�\����
struct Classifier
{
	vector<int> HistDim;						//�����ʂ̎���
	vector<double> Lut;							//�m�����x���z
	Classifier(){
		HistDim = vector<int>(PROB_DIM);
		Lut		= vector<double>(PREDICT_NUM);
	}
};

typedef	struct	OC_OBJ_INF
{
	POS3D_FLT	sMax;
	POS3D_FLT	sMin;
	POS3D_FLT	sGrav;
}	FAR* LPOC_OBJ_INF;

class	CObjAnalyze;
//////////////////////////////////////////////////////////////////////////////////////////////////
// ���ʃN���X
class CObjClassify
{
private:
	// --- �����o�ϐ�
	vector<float>			m_vecFeature;
	vector<double>			m_vecClScore;
	vector<double>			m_vecClProb;

	vector<int>				m_vecFtrList;
	vector<Classifier>		m_vecWeakCl;
	vector<vector<float>>	m_vecSplitPoint;
	vector<vector<double>>	m_vecSgmdParam;

public:
	// --- �����o�֐�
	// �\�z�Ə���
	CObjClassify();
	~CObjClassify();

	// ������
	void			Init(string& strClssfrFlNm, string& strBinFlNm, string& strSgmdPrmFlNm);

	BOOL			AnalyzeFeature(vector<POSINT_FLT>& vecPntCld, POSE_DBL sPose, CObjAnalyze& cObj, CObjTrck& cTrck);
	void			ClassifyRAB(vector<int>& vecFtrList, vector<Classifier>& vecWeakCl, vector<vector<float>>& vecSplitPoint, vector<vector<double>>& vecSgmdParam);
//	static int		UpdateClass(vector<double> vecClProb);
	static BOOL		FuncClassiy(vector<double>& vecProb, POSE_DBL sPose, CObjTrck& cTrck, CObjAnalyze& cObj, vector<POSINT_FLT>& vecPntCld, LPVOID lpData){		
			return	((CObjClassify*)lpData)->FuncClassiy(vecProb, sPose, cTrck, cObj, vecPntCld);		
	};

	// �v�Z
	UINT			CompLernNum(string& strClssfrFlNm);


	// ���擾
	vector<double>&			GetClScore(){		return	m_vecClScore;		};
	vector<double>&			GetClProb(){		return	m_vecClProb;		};
	vector<float>&			GetFeature(){		return	m_vecFeature;		};
	vector<int>&			GetFtrList(){		return	m_vecFtrList;		};
	vector<Classifier>&		GetWeakCl(){		return	m_vecWeakCl;		};
	vector<vector<float>>&	GetSplitPoint(){	return	m_vecSplitPoint;	};
	vector<vector<double>>&	GetSigmoidParam(){	return	m_vecSgmdParam;		};

private:
	// �����g�p�֐�
	void	CreateFtrStrngth(vector<float>& vecClFeature, vector<vector<UINT>>& vecFtrStrngth,  vector<vector<float>>& vecSplitPoint);
	void	Prediction(vector<vector<UINT>>& vecFtrStrngth, vector<Classifier>& vecWeakCl);
	void	Score2Prob(vector<double>& vecClScore, vector<vector<double>>& vecSgmdParam);

	// �����ʌv�Z
	void	CompObjInf(LPOC_OBJ_INF lpObjInf, vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj);		// ��SetParam�֐�
	void	CompSizeFtr(OC_OBJ_INF sObjInf);
	void	AnalyzeIntensity(vector<float>& vecIntValue);
	void	AnalyzeConvexHull(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj);
	void	AnalyzeConvexHull(CObjAnalyze& cObj);
	void	AnalyzeBinImg(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, OC_OBJ_INF sObjInf);
	void	CompCovMtrx(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, OC_OBJ_INF sObjInf);
	void	AnalyzePntHist(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, OC_OBJ_INF sObjInf);
	void	CompPntHist(vector<int>& vecPntHist, vector<float>& vecPnt, float fGrav, float fBlckSize, float fWdthRng);
	void	AnalyzeIntHist(vector<float>& vecIntValue);
	void	AnalyzeSlice(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, vector<float>& vecIntValue);
	void	CompLayer(vector<float>& vecHghtPnt, vector<int>& vecPartNum);
	void	CompPartWdth(vector<float>& vecPntCld, vector<float>& vecPartWdth, vector<int>& vecPartNum);
	void	AnalyzePartInt(vector<float>& vecIntValue, vector<float>& vecAvrgInt, vector<int>& vecPartNum);
	void	CompMultiFtr();

	// ���ʎ��s
	BOOL	FuncClassiy(vector<double>& vecProb, POSE_DBL sPose, CObjTrck& cTrck, CObjAnalyze& cObj, vector<POSINT_FLT>& vecPntCld);

	// �t�@�C���Ǎ�
	void	ReadCl_RAB(vector<Classifier>& vecWeakCl, vector<vector<float>>& vecSplitPoint, vector<vector<double>>& vecSgmdParam, string& strClssfrFlNm, string& strBinFlNm, string& strSgmdPrmFlNm);
};


inline	CObjClassify::CObjClassify()
{
	m_vecClProb	= vector<double>(CLASS_NUM);
}

inline	CObjClassify::~CObjClassify()
{
}

inline void CObjClassify::Init(string& strClssfrFlNm, string& strBinFlNm, string& strSgmdPrmFlNm)
{
	vector<int> vecFtrList;
//	UINT		nLearnNum;
	UINT		nDimNum;

	///////////////////////////////////////Track�p���ʊ�///////////////////////////////////////
	// �����ʃ��X�g
	vecFtrList.resize(DIM_NUM);
	for(int nDim = 0; nDim < DIM_NUM; ++nDim){
		vecFtrList[nDim] = nDim;
	}
	nDimNum			= vecFtrList.size();
//	nLearnNum		= LEARN_NUM_TRCK;
	
	//�������m��
//	m_vecWeakCl		= vector<Classifier>(nLearnNum * CLASS_NUM);
//	m_vecSplitPoint	= vector<vector<float>>(nDimNum, vector<float>(BIN[0]+BIN[1]));
//	m_vecSgmdParam	= vector<vector<double>>(CLASS_NUM, vector<double>(2));

	//�ǂݍ���
	ReadCl_RAB(m_vecWeakCl, m_vecSplitPoint, m_vecSgmdParam, strClssfrFlNm, strBinFlNm, strSgmdPrmFlNm);
	
	//�ۑ�
	m_vecFtrList.swap(vecFtrList);
}

inline	UINT	CObjClassify::CompLernNum(string& strClssfrFlNm)
{
	CFile	cFile;
	if( !cFile.Open(strClssfrFlNm.c_str(), CFile::modeRead|CFile::typeBinary) )
		return	(UINT)-1;

	DWORD	dwLength	= cFile.GetLength();
	UINT	nEachSize	= CLASS_NUM * ( sizeof(int)*PROB_DIM + sizeof(double)*PREDICT_NUM );
	UINT	nLernNum	= dwLength / nEachSize;

	return	nLernNum;
}

inline void CObjClassify::ReadCl_RAB(vector<Classifier>& vecWeakCl, vector<vector<float>>& vecSplitPoint, vector<vector<double>>& vecSgmdParam, string& strClssfrFlNm, string& strBinFlNm, string& strSgmdPrmFlNm)
{
	// �w�K�񐔂��t�@�C���T�C�Y����v�Z
	UINT	nLearnNum	= CompLernNum(strClssfrFlNm);

	//�������m��
	vecWeakCl		= vector<Classifier>(nLearnNum * CLASS_NUM);
	vecSplitPoint	= vector<vector<float>>(DIM_NUM, vector<float>(BIN[0]+BIN[1]));
	vecSgmdParam	= vector<vector<double>>(CLASS_NUM, vector<double>(2));

	// �㎯�ʊ�
	fstream ReadCl;
	ReadCl.open(strClssfrFlNm, ios::in | ios::binary);
	if(ReadCl.is_open())
	{
		vector<int>		HistDim(PROB_DIM);
		vector<double>	Lut(PREDICT_NUM);
//		UINT nLearnNum = vecWeakCl.size() / CLASS_NUM;
		
		for(UINT nRound = 0; nRound < nLearnNum; ++nRound){
			for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass){
				ReadCl.read((char*)&HistDim[0], sizeof(int)*PROB_DIM);
				ReadCl.read((char*)&Lut[0], sizeof(double)*PREDICT_NUM);

				vecWeakCl[nRound*CLASS_NUM + nClass].HistDim	= HistDim;
				vecWeakCl[nRound*CLASS_NUM + nClass].Lut		= Lut;
			}
		}
		ReadCl.close();
	}

	// BIN�̕�
	fstream ReadBin;
	ReadBin.open(strBinFlNm, ios::in | ios::binary);
	if (ReadBin.is_open()){
		//�����ʎ�����
		UINT nDimNum = vecSplitPoint.size();
		for(UINT nDim = 0; nDim < nDimNum; ++nDim){
			ReadBin.read((char*)&vecSplitPoint[nDim][0], sizeof(float)*(BIN[0]+BIN[1]));
		}
		ReadBin.close();
	}

	// �V�O���C�h�֐��̃p�����[�^
	fstream ReadSigmoid;
	ReadSigmoid.open(strSgmdPrmFlNm, ios::in | ios::binary);
	if (ReadSigmoid.is_open()){
		for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass){
			ReadSigmoid.read((char*)&vecSgmdParam[nClass][0], sizeof(double)*2);
		}
		ReadSigmoid.close();
	}
}

inline	BOOL	CObjClassify::AnalyzeFeature(vector<POSINT_FLT>& vecPntCld, POSE_DBL sPose, CObjAnalyze& cObj, CObjTrck& cTrck)
{
	// �������m��
	m_vecFeature = vector<float>(ALLFTR_NUM);

	// �ǐՕ��̏��̎擾
	float	fVelocity	= cTrck.CompVelocity();		// ���̑��x
	float	fVelVar		= cTrck.CompVelVar();		// ���x���U
	float	fPsi		= cTrck.GetTrckAngle();		// ���̕���

	//�|�C���g��
	UINT	pnt_size		= vecPntCld.size();
	m_vecFeature[POINTNUM]	= (float)pnt_size;

	//���ˋ��x��0-1����0-255�ɕϊ�
	vector<float> vecIntValue(pnt_size);
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecIntValue[nIndex] = 255.0f * vecPntCld[nIndex].i;
	}
	
	//Track�̒��S�p��
	CPoseDbl	cTrckPose(cTrck.GetTrckPose());
	cTrckPose.Abs2Rtv(sPose);
	POSE_FLT	sTrckPose = ConvPose(cTrckPose);

	//�|�C���g�N���E�h�̍��W��]
	vector<float>	vecLngPnt(pnt_size);
	vector<float>	vecNrrwPnt(pnt_size);
	vector<float>	vecHghtPnt(pnt_size);
	float			fCos	= cos(-sTrckPose.psi);
	float			fSin	= sin(-sTrckPose.psi);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecLngPnt[nIndex]	= (vecPntCld[nIndex].x - sTrckPose.x) * fCos - (vecPntCld[nIndex].y - sTrckPose.y) * fSin;
		vecNrrwPnt[nIndex]	= (vecPntCld[nIndex].x - sTrckPose.x) * fSin + (vecPntCld[nIndex].y - sTrckPose.y) * fCos;
		vecHghtPnt[nIndex]	= vecPntCld[nIndex].z;
	}

	// ���̏��̗\���v�Z
	OC_OBJ_INF	sObjInf;
	CompObjInf(&sObjInf, vecLngPnt, vecNrrwPnt, vecHghtPnt, cObj);

	//�傫���Ɋւ������
	CompSizeFtr(sObjInf);
	//���ˋ��x�̕��ρC���U
	AnalyzeIntensity(vecIntValue);
	//�`��Ɋւ�������ʁi�ʕ��3�ʁj
	AnalyzeConvexHull(vecLngPnt, vecNrrwPnt, vecHghtPnt, cObj);
	//Box��񂩂��ʓʕ�̂̂ݗ��p
//	AnalyzeConvexHull(cObj);
	//2�l�摜�ɂ�����ʐ�
	AnalyzeBinImg(vecLngPnt, vecNrrwPnt, vecHghtPnt, sObjInf);
	//3���������U�s��C3�����������[�����g�s��
	CompCovMtrx(vecLngPnt, vecNrrwPnt, vecHghtPnt, sObjInf);
	//�|�C���g�N���E�h���z
	AnalyzePntHist(vecLngPnt, vecNrrwPnt, sObjInf);
	//���ˋ��x���z
	AnalyzeIntHist(vecIntValue);
	//�X���C�X������
	AnalyzeSlice(vecLngPnt, vecNrrwPnt, vecHghtPnt, vecIntValue);
	//�|�����킹
	CompMultiFtr();

	// �ԑ�
	if( cTrck.GetTrckState()!=CObjTrck::stateConfirmed)
		m_vecFeature[VELOCITY]	= 0;
	else
		m_vecFeature[VELOCITY]	= fVelocity;

	//m_vecFeature[VELVAR]	= fVelVar;
	m_vecFeature[VELVAR]	= POW_SQ(fVelocity)/fVelVar;		// ��

	// �ǉ�	// ��
	RC_RECT	sRect		= cObj.GetMinAreaRect();
	float	fLength		= max(sRect.sSize.x, sRect.sSize.y);
	float	fWidth		= min(sRect.sSize.x, sRect.sSize.y);
	float	fDltHeight	= cObj.GetMaxZ() - cObj.GetMinZ();
	m_vecFeature[BOXAREA_TP]	= fLength	* fWidth;
	m_vecFeature[MINBOXVOL]		= fLength	* fWidth * fDltHeight;

	return	TRUE;
}

inline	void	CObjClassify::CompObjInf(LPOC_OBJ_INF lpObjInf, vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj)
{
	UINT pnt_size = vecLngPnt.size();

	// X,Y�̍ő�E�ŏ��l
	POS_FLT	sMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT sMin = CPosFlt(FLT_MAX, FLT_MAX);
	for( UINT nIndex=0; nIndex<pnt_size; nIndex++ )
	{
		sMax.x	= max( sMax.x, vecLngPnt[nIndex]	);
		sMax.y	= max( sMax.y, vecNrrwPnt[nIndex]	);
		sMin.x	= min( sMin.x, vecLngPnt[nIndex]	);
		sMin.y	= min( sMin.y, vecNrrwPnt[nIndex]	);
	}

	// �_�Q�̏d�S
	POS3D_FLT	sGrav;
	sGrav.x = sGrav.y = sGrav.z = 0.0f;
	for( UINT nIndex=0; nIndex<pnt_size; nIndex++ )
	{
		sGrav.x += vecLngPnt[nIndex];
		sGrav.y += vecNrrwPnt[nIndex];
		sGrav.z += vecHghtPnt[nIndex];
	}
	sGrav.x	/= (float)pnt_size;
	sGrav.y	/= (float)pnt_size;
	sGrav.z	/= (float)pnt_size;

	//�ۑ�
	lpObjInf->sMax.x	= sMax.x;
	lpObjInf->sMax.y	= sMax.y;
	lpObjInf->sMax.z	= cObj.GetMaxZ();
	lpObjInf->sMin.x	= sMin.x;
	lpObjInf->sMin.y	= sMin.y;
	lpObjInf->sMin.z	= cObj.GetMinZ();
	lpObjInf->sGrav		= sGrav;
}

inline	void	CObjClassify::CompSizeFtr(OC_OBJ_INF sObjInf)
{
	// ���E����
	m_vecFeature[LONGSIDE]		= sObjInf.sMax.x - sObjInf.sMin.x;
	m_vecFeature[NARROWSIDE]	= sObjInf.sMax.y - sObjInf.sMin.y;
	m_vecFeature[HEIGHT]		= sObjInf.sMax.z - sObjInf.sMin.z;

	// �ő�E�ŏ�����
	m_vecFeature[MAXHEIGHT]		= sObjInf.sMax.z;
	m_vecFeature[MINHEIGHT]		= sObjInf.sMin.z;
}

inline	void	CObjClassify::AnalyzeIntensity(vector<float>& vecIntValue)
{
	UINT	pnt_size = vecIntValue.size();

	float	fValue;
	float	fAvgInt	= 0.0f;
	float	fSqInt  = 0.0f;
	float	fMaxInt = -FLT_MAX;
	float	fMinInt = FLT_MAX;
	UINT	count	= 0;

	for (UINT nIndex = 0; nIndex < pnt_size-1; ++nIndex)
	{
		fValue = vecIntValue[nIndex];
		//���ˋ��x���L���Ȃ��̂������g�p
		if(fValue > 0.0f){
			fAvgInt += fValue;
			fSqInt  += fValue * fValue;
			if(fValue > fMaxInt)
				fMaxInt = fValue;
			if(fValue < fMinInt)
				fMinInt = fValue;
			++count;
		}
	}
	float	fVarInt = 0.0f;
	if(count > 0){
		//���ϒl
		fAvgInt = fAvgInt/(float)count;
		//���U�l
		fVarInt = fSqInt/(float)count - fAvgInt*fAvgInt;
	}

	m_vecFeature[INTENSITY]		= fAvgInt;
	m_vecFeature[VARINTENSITY]	= fVarInt;

	//�ő�E�ŏ��l
	m_vecFeature[INTMAX]		= fMaxInt;
	m_vecFeature[INTMIN]		= fMinInt;
}

inline	void	CObjClassify::AnalyzeConvexHull(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj)
{
	UINT	pnt_size = vecLngPnt.size();

	//OpennCV�p�Ɍ^�ϊ�
	vector<Point2f>	vecSdCvPnts(pnt_size);
	vector<Point2f>	vecFrntCvPnts(pnt_size);
	for (UINT nLoop = 0; nLoop < pnt_size; nLoop++)
	{
		vecSdCvPnts[nLoop]		= Point2f(vecLngPnt[nLoop], vecHghtPnt[nLoop]);
		vecFrntCvPnts[nLoop]	= Point2f(vecNrrwPnt[nLoop], vecHghtPnt[nLoop]);
	}
	//���ʁE����
	vector<Point2f>		vecSdConvexHull;
	vector<Point2f>		vecFrntConvexHull;
	convexHull(vecSdCvPnts, vecSdConvexHull);
	convexHull(vecFrntCvPnts, vecFrntConvexHull);

	//��ʂ�Box���𗘗p
	vector<Point2f>		vecTpConvexHull;
	vector<POS_FLT>&	vecContour = cObj.GetContour();
	vecTpConvexHull.reserve(vecContour.size());
	vector<POS_FLT>::iterator	itrContour;
	for( itrContour=vecContour.begin(); itrContour!=vecContour.end(); itrContour++ )
		vecTpConvexHull.push_back( cObj.Conv(*itrContour) );

	//�ʐ�
	double SdArea		= contourArea(vecSdConvexHull);
	double FrntArea		= contourArea(vecFrntConvexHull);
	double TpArea		= contourArea(vecTpConvexHull);
	//���͒���
	double SdLength		= arcLength(vecSdConvexHull,true);
	double FrntLength	= arcLength(vecFrntConvexHull,true);
	double TpLength		= arcLength(vecTpConvexHull,true);

	m_vecFeature[SIDEAREA]		= (float)SdArea;
	m_vecFeature[FRONTAREA]		= (float)FrntArea;
	m_vecFeature[TOPAREA]		= (float)TpArea;

	m_vecFeature[SIDEPER]		= (float)SdLength;
	m_vecFeature[FRONTPER]		= (float)FrntLength;
	m_vecFeature[TOPPER]		= (float)TpLength;
}

inline	void	CObjClassify::AnalyzeConvexHull(CObjAnalyze& cObj)
{
	//��ʂ�Box���𗘗p
	vector<Point2f>		vecTpConvexHull;
	vector<POS_FLT>&	vecContour = cObj.GetContour();
	vecTpConvexHull.reserve(vecContour.size());
	vector<POS_FLT>::iterator	itrContour;
	for( itrContour=vecContour.begin(); itrContour!=vecContour.end(); itrContour++ )
		vecTpConvexHull.push_back( cObj.Conv(*itrContour) );

	//�ʐ�
	double TpArea		= contourArea(vecTpConvexHull);
	//���͒���
	double TpLength		= arcLength(vecTpConvexHull,true);

	m_vecFeature[TOPAREA]		= (float)TpArea;
	m_vecFeature[TOPPER]		= (float)TpLength;
}

inline	void	CObjClassify::AnalyzeBinImg(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, OC_OBJ_INF sObjInf)
{
	UINT	pnt_size	= vecLngPnt.size();

	//X,Y,Z�̒��S���W
	POS3D_FLT obCenter;
	obCenter.x = 0.5f*(sObjInf.sMax.x	+ sObjInf.sMin.x);
	obCenter.y = 0.5f*(sObjInf.sMax.y	+ sObjInf.sMin.y);
	obCenter.z = 0.5f*(sObjInf.sMax.z	+ sObjInf.sMin.z);
	//�E�B���h�E�T�C�Y(float)
	POS3D_FLT fWindSize;
	fWindSize.x = (sObjInf.sMax.x	- sObjInf.sMin.x)	/ PIXEL_SIZE + 10.0f;
	fWindSize.y = (sObjInf.sMax.y	- sObjInf.sMin.y)	/ PIXEL_SIZE + 10.0f;
	fWindSize.z = (sObjInf.sMax.z	- sObjInf.sMin.z)	/ PIXEL_SIZE + 10.0f;
	//�E�B���h�E�T�C�Y(int)
	int iWindSizeX = int(0.5f+fWindSize.x);
	int iWindSizeY = int(0.5f+fWindSize.y);
	int iWindSizeZ = int(0.5f+fWindSize.z);
	//���ʁE���ʁE��ʉ摜����
	cv::Mat Side_img	= cv::Mat::zeros(cv::Size(iWindSizeX, iWindSizeZ), CV_8UC1);
	cv::Mat Front_img	= cv::Mat::zeros(cv::Size(iWindSizeY, iWindSizeZ), CV_8UC1);
	cv::Mat Top_img		= cv::Mat::zeros(cv::Size(iWindSizeX, iWindSizeY), CV_8UC1); 
	//�摜���S
	POS3D_FLT pxCenter;
	pxCenter.x = 0.5f * fWindSize.x;
	pxCenter.y = 0.5f * fWindSize.y;
	pxCenter.z = 0.5f * fWindSize.z;

	POS3D_FLT delta;
	int iPixX, iPixY1, iPixY2, iPixZ;
	for (UINT i = 0; i < pnt_size; ++i){
		//���S���W����̍�
		delta.x = (vecLngPnt[i]		- obCenter.x) / PIXEL_SIZE;
		delta.y = (vecNrrwPnt[i]	- obCenter.y) / PIXEL_SIZE;
		delta.z = (vecHghtPnt[i]	- obCenter.z) / PIXEL_SIZE;
		//�摜��̍��W
		iPixX	= int(pxCenter.x + delta.x);
		iPixY1	= int(pxCenter.y + delta.y);
		iPixY2	= int(pxCenter.y - delta.y);
		iPixZ	= int(pxCenter.z - delta.z);
		//���e
		Side_img.at<uchar>(iPixZ, iPixX)	= 255;
		Front_img.at<uchar>(iPixZ, iPixY1)	= 255;
		Top_img.at<uchar>(iPixY2, iPixX)	= 255;
	}
	//���e���ꂽ�̈���v�Z�i�ʐρj
	int SideArea	= countNonZero(Side_img);
	int FrontArea	= countNonZero(Front_img);
	int TopArea		= countNonZero(Top_img);

	m_vecFeature[SD_NONZERO]	= (float)SideArea;
	m_vecFeature[FRNT_NONZERO]	= (float)FrontArea;
	m_vecFeature[TP_NONZERO]	= (float)TopArea;
}

inline	void	CObjClassify::CompCovMtrx(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, OC_OBJ_INF sObjInf)
{
	UINT		pnt_size	= vecLngPnt.size();
	POS3D_FLT	sGrav		= sObjInf.sGrav;	// GetPntGrav();

	POS3D_FLT sPosDlt;
	vector<float> vecCovElmnt(COVMTRX_ELNUM, 0.0f);		//3���������U�s��
	vector<float> vecInrElmnt(INRMTRX_ELNUM, 0.0f);		//3�����������[�����g�s��
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		//�d�S�𒆐S�Ƃ������W
		sPosDlt.x = vecLngPnt[nIndex]	- sGrav.x;
		sPosDlt.y = vecNrrwPnt[nIndex]	- sGrav.y;
		sPosDlt.z = vecHghtPnt[nIndex]	- sGrav.z;
		//3���������U�s��
		vecCovElmnt[0] += sPosDlt.x * sPosDlt.x;
		vecCovElmnt[1] += sPosDlt.x * sPosDlt.y;
		vecCovElmnt[2] += sPosDlt.x * sPosDlt.z;
		vecCovElmnt[3] += sPosDlt.y * sPosDlt.y;
		vecCovElmnt[4] += sPosDlt.y * sPosDlt.z;
		vecCovElmnt[5] += sPosDlt.z * sPosDlt.z;
		//3�����������[�����g�s��
		vecInrElmnt[0] += sPosDlt.y * sPosDlt.y + sPosDlt.z * sPosDlt.z;
		vecInrElmnt[1] -= sPosDlt.x * sPosDlt.y;
		vecInrElmnt[2] -= sPosDlt.x * sPosDlt.z;
		vecInrElmnt[3] += sPosDlt.x * sPosDlt.x + sPosDlt.z * sPosDlt.z;
		vecInrElmnt[4] -= sPosDlt.y * sPosDlt.z;
		vecInrElmnt[5] += sPosDlt.x * sPosDlt.x + sPosDlt.y * sPosDlt.y;
	}
	//3���������U�s��̂ݐ��K��
	for(UINT i = 0; i < COVMTRX_ELNUM; ++i){
		vecCovElmnt[i] /= (float)pnt_size;
	}

	for(UINT j = 0; j < COVMTRX_ELNUM; ++j){
		m_vecFeature[COVMTRXSTRT+j] = vecCovElmnt[j];
	}
	for(UINT k = 0; k < INRMTRX_ELNUM; ++k){
		m_vecFeature[INRMTRXSTRT+k] = vecInrElmnt[k];
	}
}

inline	void	CObjClassify::AnalyzePntHist(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, OC_OBJ_INF sObjInf)
{
	UINT pnt_size = vecLngPnt.size();

	float fLngGrav	= sObjInf.sGrav.x;
	float fNrrwGrav = sObjInf.sGrav.y;
	
	float		fBlckSize	= PNTHST_PTCH;
	float		fWdthRng	= PNTHST_LNGRNG;
	vector<int> vecLngPntHist;
	//���ʃ|�C���g�N���E�h���z(�q�X�g�O����)
	CompPntHist(vecLngPntHist, vecLngPnt, fLngGrav, fBlckSize, fWdthRng);

	fBlckSize	= PNTHST_PTCH;
	fWdthRng	= PNTHST_NRRWRNG;
	vector<int> vecNrrwPntHist;
	//���ʃ|�C���g�N���E�h���z(�q�X�g�O����)
	CompPntHist(vecNrrwPntHist, vecNrrwPnt, fNrrwGrav, fBlckSize, fWdthRng);

	//���K�����ĕۑ�
	for(UINT i = 0; i < vecLngPntHist.size(); ++i){
		m_vecFeature[PNTHSTLNGSTRT+i]	= (float)vecLngPntHist[i]	/ (float)pnt_size;
	}
	for(UINT j = 0; j < vecNrrwPntHist.size(); ++j){
		m_vecFeature[PNTHSTNRRWSTRT+j]	= (float)vecNrrwPntHist[j]	/ (float)pnt_size;
	}
}

inline	void	CObjClassify::CompPntHist(vector<int>& vecPntHist, vector<float>& vecPnt, float fGrav, float fBlckSize, float fWdthRng)
{
	UINT		pnt_size	= vecPnt.size();

	int			nLoopNum	= int(0.5 + fWdthRng/fBlckSize);	//��������������(�l�̌ܓ�)
	int			tmp			= int(0.5 * nLoopNum);
	//�����_�ŏ��l
	float		fMinHrz		= fGrav - 0.5f*fBlckSize - fBlckSize*(float)tmp;		

	int			num;
	float		fDlt, fRate;
	vector<int>	vecHist(nLoopNum, 0);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		//�ŏ��l�Ƃ̍�
		fDlt	= vecPnt[nIndex] - fMinHrz;
		//�S���ɑ΂��銄��0-1
		fRate	= fDlt	/ fWdthRng;
		//BIN�̔ԍ�
		num		= int(fRate * nLoopNum);

		if(num < 0)
			num = 0;
		else if(num > nLoopNum-1)
			num = nLoopNum -1;

		++vecHist[num];
	}

	vecPntHist.swap(vecHist);
}

inline	void	CObjClassify::AnalyzeIntHist(vector<float>& vecIntValue)
{
	UINT			pnt_size = vecIntValue.size();

	float			fMaxVal		= 256.0f;
	//BIN��(�l�̌ܓ�)
	int				nLoopNum	= INTHST_DIM;

	int				num;
	UINT			nCnt = 0;
	float			fValue;
	float			fRate;
	vector<UINT>	vecIntHist(nLoopNum, 0);
	for (UINT nIndex = 0; nIndex < pnt_size-1; ++nIndex){
		fValue = vecIntValue[nIndex];
		//���ˋ��x���L���Ȃ��̂������g�p
		if(fValue > 0.0f){
			//�S���ɑ΂��銄��0-1
			fRate	= fValue / fMaxVal;
			//BIN�̔ԍ�
			num		= int(fRate * nLoopNum);
			if(num > nLoopNum-1)
				num = nLoopNum -1;

			++vecIntHist[num];
			++nCnt;
		}
	}

	if(nCnt > 0){
		//���K��
		for(int nLoop = 0; nLoop < nLoopNum; ++nLoop){
			m_vecFeature[INTHSTSTRT+nLoop] = (float)vecIntHist[nLoop] / (float)nCnt;
		}
	}else{
		for(int nLoop = 0; nLoop < nLoopNum; ++nLoop){
			m_vecFeature[INTHSTSTRT+nLoop] = (float)vecIntHist[nLoop];	
		}
	}
}

inline	void	CObjClassify::AnalyzeSlice(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, vector<float>& vecIntValue)
{
	UINT nLoopNum = LAYERNUM;

	vector<int>		vecPartNum;
	//�|�C���g�N���E�h�𕪊�
	CompLayer(vecHghtPnt, vecPartNum);
	//�p�[�c���̕�(����)
	vector<float>	vecLngWdth(nLoopNum);
	CompPartWdth(vecLngPnt, vecLngWdth, vecPartNum);
	//�p�[�c���̕�(�Z��)
	vector<float>	vecNrrwWdth(nLoopNum);
	CompPartWdth(vecNrrwPnt, vecNrrwWdth, vecPartNum);
	//�p�[�c���̕��ϔ��ˋ��x
	vector<float>	vecAvrgInt(nLoopNum, 0.0f);
	AnalyzePartInt(vecIntValue, vecAvrgInt, vecPartNum);

	float	fPartAvrgInt	= 0.0f;	//�S�p�[�c�ɂ����镽�ϔ��ˋ��x
	float	fPartSqInt		= 0.0f;	
	float	fPartVarInt		= 0.0f;	//�S�p�[�c�ɂ����锽�ˋ��x���U
	float	fValue;
	UINT	nCnt			= 0;
	for(UINT nLoop = 0; nLoop < nLoopNum; ++nLoop){
		fValue = vecAvrgInt[nLoop];
		//���ˋ��x���L���Ȃ��̂������g�p
		if(fValue > 0.0f){
			fPartAvrgInt	+= fValue;
			fPartSqInt		+= fValue * fValue;
			++nCnt;
		}
	}
	if(nCnt > 0){
		//���ϒl
		fPartAvrgInt /= (float)nCnt;
		//���U�l
		fPartVarInt = fPartSqInt/(float)nCnt - fPartAvrgInt*fPartAvrgInt;
	}

	for(UINT nLoop = 0; nLoop < nLoopNum; ++nLoop){
		m_vecFeature[SLICE_LWSTRT + nLoop] = vecLngWdth[nLoop];
		m_vecFeature[SLICE_NWSTRT + nLoop] = vecNrrwWdth[nLoop];
		m_vecFeature[SLICE_AISTRT + nLoop] = vecAvrgInt[nLoop];
	}
	m_vecFeature[SLICE_PAI] = fPartAvrgInt;
	m_vecFeature[SLICE_PVI] = fPartVarInt;
}

inline	void	CObjClassify::CompLayer(vector<float>& vecHghtPnt, vector<int>& vecPartNum)
{
	UINT		pnt_size = vecHghtPnt.size();
	vector<int> vecLayerNum(pnt_size);

	UINT	nLoopNum	= LAYERNUM;						//���C���[�̐�
	float	fWdith		= SLICE_MAXHGHT-SLICE_MINHGHT;	//�S��
	float	fMinHght	= SLICE_MINHGHT;				//�����̍ŏ��l
	
	int		num;
	float	fDlt;
	float	fRate;
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		//�ŏ��l�Ƃ̍�
		fDlt	= vecHghtPnt[nIndex] - fMinHght;
		//�S���ɑ΂��銄��0-1
		fRate	= fDlt / fWdith;
		//���C���[�̔ԍ�
		num = (int)(fRate * nLoopNum);
		if(num < 0)
			num = 0;
		else if(num > (int)nLoopNum-1)
			num = nLoopNum -1;

		vecLayerNum[nIndex] = num;
	}

	vecPartNum.swap(vecLayerNum);
}

inline	void	CObjClassify::CompPartWdth(vector<float>& vecPntCld, vector<float>& vecPartWdth, vector<int>& vecPartNum)
{
	UINT pnt_size	= vecPntCld.size();
	UINT nLoopNum	= vecPartWdth.size();			//���C���[�̐�
	vector<float> vecPartMax(nLoopNum, -FLT_MAX);	//�e���C���[�ɂ�����ő�l
	vector<float> vecPartMin(nLoopNum, FLT_MAX);	//�e���C���[�ɂ�����ŏ��l
	
	int nPart;
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		//���C���[�̔ԍ�
		nPart = vecPartNum[nIndex];
		//�e���C���[�ɂ�����ő�E�ŏ��l
		if(vecPntCld[nIndex] > vecPartMax[nPart])
			vecPartMax[nPart] = vecPntCld[nIndex];
		if(vecPntCld[nIndex] < vecPartMin[nPart])
			vecPartMin[nPart] = vecPntCld[nIndex];
	}
	//�e���C���[�̕�
	for(UINT nLoop = 0; nLoop < nLoopNum; ++nLoop){
		if(vecPartMax[nLoop] > vecPartMin[nLoop]){
			vecPartWdth[nLoop] = vecPartMax[nLoop] - vecPartMin[nLoop];
		}else{
			vecPartWdth[nLoop] = 0.0f;
		}
	}
}

inline	void	CObjClassify::AnalyzePartInt(vector<float>& vecIntValue, vector<float>& vecAvrgInt, vector<int>& vecPartNum)
{
	UINT			pnt_size	= vecIntValue.size();
	UINT			nLoopNum	= vecAvrgInt.size();	//���C���[�̐�
	vector<UINT>	vecCount(nLoopNum, 0);				//�L���Ȕ��ˋ��x�����|�C���g�̐�

	int nPart;
	for(UINT nIndex = 0; nIndex < pnt_size-1; ++nIndex){
		//�L���Ȕ��ˋ��x�̂ݎg�p
		if(vecIntValue[nIndex] > 0.0f){
			//���C���[�̔ԍ�
			nPart = vecPartNum[nIndex];

			vecAvrgInt[nPart] += vecIntValue[nIndex];
			++vecCount[nPart];
		}
	}
	//�e���C���[�̕��ϔ��ˋ��x
	for(UINT nLoop = 0; nLoop < nLoopNum; ++nLoop){
		if(vecCount[nLoop] > 0){
			vecAvrgInt[nLoop] /= (float)vecCount[nLoop];
		}
	}
}

inline	void	CObjClassify::CompMultiFtr()
{
	float epsilon_two	= 0.01f;
	float epsilon_three = 0.001f;
	float fLngSd, fNrrwSd, fHght, fPntNm, fInt, fVar, fSdA, fFrntA, fTpA, fSdPr, fFrntPr, fTpPr, fSdNonZ, fFrntNonZ, fTpNonZ;

	fLngSd		= m_vecFeature[LONGSIDE];
	fNrrwSd		= m_vecFeature[NARROWSIDE];
	fHght		= m_vecFeature[HEIGHT];
	fPntNm		= m_vecFeature[POINTNUM];
	fInt		= m_vecFeature[INTENSITY];
	fVar		= m_vecFeature[VARINTENSITY];
	fSdA		= m_vecFeature[SIDEAREA];
	fFrntA		= m_vecFeature[FRONTAREA];
	fTpA		= m_vecFeature[TOPAREA];
	fSdPr		= m_vecFeature[SIDEPER];
	fFrntPr		= m_vecFeature[FRONTPER];
	fTpPr		= m_vecFeature[TOPPER];
	fSdNonZ		= m_vecFeature[SD_NONZERO];
	fFrntNonZ	= m_vecFeature[FRNT_NONZERO];
	fTpNonZ		= m_vecFeature[TP_NONZERO];

	m_vecFeature[VOLBOX]			= fLngSd	* fNrrwSd	* fHght;
	m_vecFeature[MUL_LNGNRRW]		= fLngSd	* fNrrwSd;
	m_vecFeature[MUL_LNGHGHT]		= fLngSd	* fHght;
	m_vecFeature[DIV_VARINT]		= fVar		/ (fInt					+ epsilon_two);
	m_vecFeature[PNTDENSITY]		= fPntNm	/ (fLngSd	* fHght		+ epsilon_two);
	m_vecFeature[DIV_HGHTLNG]		= fHght		/ (fLngSd				+ epsilon_three);
	m_vecFeature[DIV_HGHTNRRW]		= fHght		/ (fNrrwSd				+ epsilon_three);
	m_vecFeature[SDDENS]			= fSdA		/ (fLngSd	* fHght		+ epsilon_two);
	m_vecFeature[FRNTDENS]			= fFrntA	/ (fNrrwSd	* fHght		+ epsilon_two);
	m_vecFeature[TPDENS]			= fTpA		/ (fLngSd	* fNrrwSd	+ epsilon_two);
	m_vecFeature[SDCIRCLE]			= fSdA		/ (fSdPr	* fSdPr		+ epsilon_two);
	m_vecFeature[FRNTCIRCLE]		= fFrntA	/ (fFrntPr	* fFrntPr	+ epsilon_two);
	m_vecFeature[TPCIRCLE]			= fTpA		/ (fTpPr	* fTpPr		+ epsilon_two);
	m_vecFeature[SDELONG]			= fSdA		/ (fLngSd				+ epsilon_three);
	m_vecFeature[FRNTELONG]			= fFrntA	/ (fNrrwSd				+ epsilon_three);
	m_vecFeature[TPELONG]			= fTpA		/ (fLngSd				+ epsilon_three);
	m_vecFeature[SDDENS_NONZ]		= fSdNonZ	/ (fSdA					+ epsilon_three);
	m_vecFeature[FRNTDENS_NONZ]		= fFrntNonZ / (fFrntA				+ epsilon_three);
	m_vecFeature[TPDENS_NONZ]		= fTpNonZ	/ (fTpA					+ epsilon_three);
	m_vecFeature[SDELONG_NONZ]		= fSdNonZ	/ (fLngSd				+ epsilon_three);
	m_vecFeature[FRNTELONG_NONZ]	= fFrntNonZ / (fNrrwSd				+ epsilon_three);
	m_vecFeature[TPELONG_NONZ]		= fTpNonZ	/ (fLngSd				+ epsilon_three);
}

inline	void	CObjClassify::ClassifyRAB(vector<int>& vecFtrList, vector<Classifier>& vecWeakCl, vector<vector<float>>& vecSplitPoint, vector<vector<double>>& vecSgmdParam)
{
	vector<float>&	vecFeature	= GetFeature();

	//�����x�N�g���̍쐬
	int				tmp;
	UINT			nDimNum	= vecFtrList.size();	
	vector<float>	vecClFeature(nDimNum);			
	for(UINT nDim = 0; nDim < nDimNum; ++nDim){
		tmp = vecFtrList[nDim];
		vecClFeature[nDim] = vecFeature[tmp];
	}
	//�����ʂ�������x�ɕϊ�
	vector<vector<UINT>>	vecFtrStrngth;
	CreateFtrStrngth(vecClFeature, vecFtrStrngth, vecSplitPoint);

	//���ރX�R�A�̎Z�o
	Prediction(vecFtrStrngth, vecWeakCl);
	vector<double>&	vecClScore = GetClScore();

	//���ރX�R�A����m���ɕϊ�
	Score2Prob(vecClScore, vecSgmdParam);
}

inline	void	CObjClassify::CreateFtrStrngth(vector<float>& vecClFeature, vector<vector<UINT>>& vecFtrStrngth,  vector<vector<float>>& vecSplitPoint)
{
	//������
	UINT nDimNum	= vecSplitPoint.size();
	//�������m��
	vecFtrStrngth	= vector<vector<UINT>>(PROB_DIM, vector<UINT>(nDimNum));

	//�񕪒T�����������x�����߂�
	size_t tmp;
	vector<float>::iterator itrStrt;
	vector<float>::iterator itrBound;
	for(UINT i = 0; i < PROB_DIM; ++i)
	{
		for(UINT nDim = 0; nDim < nDimNum; ++nDim)
		{
			//BIN�̕��̈ꗗ
			itrStrt		= vecSplitPoint[nDim].begin() + i*BIN[0];
			//�񕪒T��
			itrBound	= upper_bound(itrStrt, itrStrt+BIN[i], vecClFeature[nDim]);
			//BIN�̔ԍ�
			tmp = distance(itrStrt, itrBound);
			if( (int)tmp > BIN[i] - 1)
				tmp = BIN[i] - 1;

			vecFtrStrngth[i][nDim] = tmp;
		}
	}
}

inline	void	CObjClassify::Prediction(vector<vector<UINT>>& vecFtrStrngth, vector<Classifier>& vecWeakCl)
{
	//�w�K��
	UINT	nLearnNum = vecWeakCl.size() / CLASS_NUM;
	
	int				iDim;
	int				iNum;
	UINT			tmp;
	UINT			nBin[PROB_DIM];
	vector<double>	vecClScore(CLASS_NUM, 0.0);
	//�w�K�񐔃��[�v
	for(UINT nRound = 0; nRound < nLearnNum; ++nRound){
		//�N���X�����[�v
		for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass){
			tmp	= nRound*CLASS_NUM + nClass;
			//��������
			vector<int>& HistDim = vecWeakCl[tmp].HistDim;

			for(UINT i = 0; i < PROB_DIM; ++i){
				//����
				iDim	= HistDim[i];
				//BIN�̔ԍ�
				nBin[i] = vecFtrStrngth[i][iDim];
			}
			iNum = nBin[1]*BIN[0] + nBin[0];

			//�X�R�A�̎Z�o
			vecClScore[nClass] += vecWeakCl[tmp].Lut[iNum];
		}
	}

	m_vecClScore.swap(vecClScore);
}

inline	void	CObjClassify::Score2Prob(vector<double>& vecClScore, vector<vector<double>>& vecSgmdParam)
{
	//�V�O���C�h�֐����m���ɕϊ�
	double dAlpha;
	for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass){
		dAlpha = exp(vecSgmdParam[nClass][0] + vecClScore[nClass]*vecSgmdParam[nClass][1]);
		m_vecClProb[nClass] = dAlpha / (1.0 + dAlpha);
	}
}

/*inline	int	CObjClassify::UpdateClass(vector<double> vecClProb)
{
	int		nObClass;
	double	dMax = -DBL_MAX;

	//�m�����ő�̃N���X�֎��ʂ���
	for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass)
	{
		if(vecClProb[nClass] > dMax){
			dMax		= vecClProb[nClass];
			nObClass	= nClass;
		}
	}

	return	nObClass;
}
*/

inline	BOOL	CObjClassify::FuncClassiy(vector<double>& vecProb, POSE_DBL sPose, CObjTrck& cTrck, CObjAnalyze& cObj, vector<POSINT_FLT>& vecPntCld)
{
	//�����ʌv�Z
	if( !AnalyzeFeature(vecPntCld, sPose, cObj, cTrck) )
		return	FALSE;

	//����-RealAdaBoost
	ClassifyRAB(GetFtrList(), GetWeakCl(), GetSplitPoint(), GetSigmoidParam());

	// �N���X�m���̕ۑ�
	vecProb	= GetClProb();

	return	TRUE;
}
