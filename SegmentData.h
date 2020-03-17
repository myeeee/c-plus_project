#pragma once 

#include "feature.h"
#include "track_manager.h"
#include "ObjAnalyze.h"
#include <iostream>
#include <vector>
#include <numeric>

using namespace	cv;

#define PI				(3.1415926535897932384626433832795)
#define PIXEL_SIZE		0.06f

const float FLTMAX_SIZE[CLASS_NUM]	= { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

const float MAX_SIZE[CLASS_NUM]		= { 6.0, 1.9/*1.05*/, 2.2, FLT_MAX };	
const float MAX_HEIGHT[CLASS_NUM]	= { 2.1, 1.9		, 2.1, FLT_MAX };	
const float MIN_POINT = 5;	
const float MAX_POINT = FLT_MAX;
//const float MAX_POINT = 20;
const float MIN_DISTANCE = 0;
//const float MAX_DISTANCE = 70;	
const float MAX_DISTANCE = FLT_MAX;	

#define COVMTRX_ELNUM	6
#define INRMTRX_ELNUM	6

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

using namespace std;

enum DATA_TYPE{
	NOTUSE, TRAINING, TEST
};

//�����ʃ��X�g
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

	BOXAREA_TP,
	// Box�̐� 
	MINBOXVOL,
	
	DIM_NUM,	

	//���Ԃ���̋���
	DISTANCE,
	
	//�n�ʂ���̍ő�E�ŏ�����
	MAXHEIGHT,
	MINHEIGHT,

	// Box�T�C�Y
	BOXSIZE_LNG,
	BOXSIZE_NRRW,
	// Box�ʐ�
	BOXAREA_SD,
	BOXAREA_FRNT,

	//�ȉ~����
	SD_ELLPSLNG,
	FRNT_ELLPSLNG,
	TP_ELLPSLNG,
	//�ȉ~�Z��
	SD_ELLPSNRRW,
	FRNT_ELLPSNRRW,
	TP_ELLPSNRRW,
	//�ȉ~����
	SD_ELLPSANGL,
	FRNT_ELLPSANGL,
	TP_ELLPSANGL,

	ALLFTR_NUM
};

////�����ʃ��X�g
//enum Feature
//{
//	//�傫��
//	LONGSIDE,
//	NARROWSIDE,
//	HEIGHT,
//	//�|�C���g��
//	POINTNUM,
//	//���ˋ��x���ρE���U�l
//	INTENSITY,
//	VARINTENSITY,
//
//	
//	//3���������U�s��
//	COVMTRXSTRT,
//	COVMTRXEND = COVMTRXSTRT + COVMTRX_ELNUM - 1,
//	//3�����������[�����g�s��
//	INRMTRXSTRT,
//	INRMTRXEND = INRMTRXSTRT + INRMTRX_ELNUM - 1,
//	
//	//���ˋ��x���z
//	INTHSTSTRT,
//	INTHSTEND = INTHSTSTRT + INTHST_DIM - 1,
//	//�ő�E�ŏ����ˋ��x
//	INTMAX,
//	INTMIN,
//	//�X���C�X������
//	SLICE_LWSTRT,
//	SLICE_LWEND = SLICE_LWSTRT + LAYERNUM - 1,
//	SLICE_NWSTRT,
//	SLICE_NWEND = SLICE_NWSTRT + LAYERNUM - 1,
//	SLICE_AISTRT,
//	SLICE_AIEND = SLICE_AISTRT + LAYERNUM - 1,
//	SLICE_PAI,
//	SLICE_PVI,
//	
//	//���x
//	VELOCITY,
//
//	
//
//	DIM_NUM,
//
//	//�ʕ�́i���ʁj
//	SIDEAREA,
//	SIDEPER,
//	//�ʕ�́i���ʁj
//	FRONTAREA,
//	FRONTPER,
//	//�ʕ�́i��ʁj
//	TOPAREA,
//	TOPPER,
//
//	//2�l�摜�ʐ�
//	SD_NONZERO,
//	FRNT_NONZERO,
//	TP_NONZERO,
//
//	//�|�C���g�N���E�h���z
//	PNTHSTLNGSTRT,
//	PNTHSTLNGEND = PNTHSTLNGSTRT + PNTHST_LNGDIM - 1,
//	PNTHSTNRRWSTRT,
//	PNTHSTNRRWEND = PNTHSTNRRWSTRT + PNTHST_NRRWDIM - 1,
//
//	
//
//	//�|�����킹
//	DIV_VARINT,
//	PNTDENSITY,
//	VOLBOX,
//	MUL_LNGNRRW,
//	MUL_LNGHGHT,
//	DIV_HGHTLNG,
//	DIV_HGHTNRRW,
//
//	SDDENS,
//	FRNTDENS,
//	TPDENS,
//
//	SDCIRCLE,
//	FRNTCIRCLE,
//	TPCIRCLE,
//
//	SDELONG,
//	FRNTELONG,
//	TPELONG,
//
//	SDDENS_NONZ,
//	FRNTDENS_NONZ,
//	TPDENS_NONZ,
//
//	SDELONG_NONZ,
//	FRNTELONG_NONZ,
//	TPELONG_NONZ,
//
//	//���x�̕��U
//	VELVAR,
//
//	BOXAREA_TP,
//	// Box�̐� 
//	MINBOXVOL,
//
//	//���Ԃ���̋���
//	DISTANCE,
//	
//	//�n�ʂ���̍ő�E�ŏ�����
//	MAXHEIGHT,
//	MINHEIGHT,
//
//	// Box�T�C�Y
//	BOXSIZE_LNG,
//	BOXSIZE_NRRW,
//	// Box�ʐ�
//	BOXAREA_SD,
//	BOXAREA_FRNT,
//
//	//�ȉ~����
//	SD_ELLPSLNG,
//	FRNT_ELLPSLNG,
//	TP_ELLPSLNG,
//	//�ȉ~�Z��
//	SD_ELLPSNRRW,
//	FRNT_ELLPSNRRW,
//	TP_ELLPSNRRW,
//	//�ȉ~����
//	SD_ELLPSANGL,
//	FRNT_ELLPSANGL,
//	TP_ELLPSANGL,
//
//	ALLFTR_NUM
//};

class CSegment
{
private:
	vector<float>	m_vecFeature;
	POS3D_FLT		m_obMax;
	POS3D_FLT		m_obMin;
	POS3D_FLT		m_obGrav;
	int				m_nDataType;
	
public:
	CSegment();
	void	SetFeature(int nFtrIndx, float fValue){	m_vecFeature[nFtrIndx] = fValue; };
	void	CompSizeFtr();
	void	AnalyzeIntensity(vector<float>& vecIntValue);
	void	EstimateDist(vector<POS3D_FLT>& vecPntCld, pose_t& Mpose);
	void	EstimateDist(vector<POS3D_FLT>& vecPntCld);
	void	AnalyzeEllipse(vector<Point2f>& vecSdCvPnts, vector<Point2f>& vecFrntCvPnts, vector<Point2f>& vecTpCvPnts);
	void	AnalyzeBinImg(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	bool	CheckSgm(int nLabel);
	void	CompMultiFtr();

	float		GetFeature(int nDim){ return m_vecFeature[nDim];	};
	POS3D_FLT	GetPntGrav(){	return	m_obGrav;					};
	int			GetDataType(){	return	m_nDataType;				};

	bool	SgmAnalyze(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntValue, pose_t& Mpose, int nLabel, POSE_FLT sTrckPose);
	bool	SgmAnalyze(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntValue, pose_t& Mpose, int nLabel);

	void	CompCovMtrx(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	void	AnalyzePntHist(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt);
	void	CompPntHist(vector<int>& vecPntHist, vector<float>& vecPnt, float fGrav, float fBlckSize, float fWdthRng);
	void	AnalyzeIntHist(vector<float>& vecIntValue);
	void	AnalyzeBoxInf(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	void	SetParam(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	void	CompRectFtr(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	void	AnalyzeConvexHull(CObjAnalyze& cObj, vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt);
	void	AnalyzeConvexHull(CObjAnalyze& cObj);
	void	AnalyzeSlice(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, vector<float>& vecIntValue);
	void	CompLayer(vector<float>& vecHghtPnt, vector<int>& vecPartNum);
	void	CompPartWdth(vector<float>& vecPntCld, vector<float>& vecPartWdth, vector<int>& vecPartNum);
	void	AnalyzePartInt(vector<float>& vecIntValue, vector<float>& vecAvrgInt, vector<int>& vecPartNum);

	void	SetDataType(int nType){		m_nDataType = nType;	}

	//�ÓI�����o�ϐ�
	static void ShowSgmImg(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntensity, float pixel_size);
	static void ShowSgmImgColor(vector<POS3D_FLT>& vecPntCld, POSE_FLT sPose, float pixel_size);

	static void RotPntCld(vector<POS3D_FLT>& vecPntCld, vector<POS3D_FLT>& vecRotPntCld);
	static void ShowFalse(vector<POS3D_FLT>& vecPntCld, POSE_FLT sPose, stringstream& sstrText, cv::Scalar& color, string& strFileName);
	static void	ImgCalar(int nLabel, cv::Scalar& color);
};

inline CSegment::CSegment()
{
	m_vecFeature	= vector<float>(ALLFTR_NUM);
	m_nDataType		= DATA_TYPE::NOTUSE;
}

inline bool CSegment::SgmAnalyze(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntValue, pose_t& Mpose, int nLabel, POSE_FLT sTrckPose)
{
	size_t	pnt_size = vecPntCld.size();
	m_vecFeature[POINTNUM] = (float)pnt_size;
	
	// ���̂̉��
	CObjAnalyze	cObject;
	if( !cObject.Analyze(vecPntCld) )
		return false;

	// ���������镨�̂̍폜(�m�C�Y�����Ɍ���C�傫�����͎̂c���Ă����Ȃ��Ƃ��Ƃœʕ�̂ɓ����Ă��镨�̂��폜�ł��Ȃ��Ȃ�)
	RC_RECT	sRect	= cObject.GetMinAreaRect();
	float	fLength	= max(sRect.sSize.x, sRect.sSize.y);	
	if( fLength<0.2 )						
		return false;

	float	fDist		= hypot(sRect.sCenter.x, sRect.sCenter.y);
	float	fDltHeight	= cObject.GetMaxZ() - cObject.GetMinZ();
	if( fDltHeight<0.5 && fDist<30 )
		return false;

	// �傫�����镨��
	float	fWidth	= min(sRect.sSize.x, sRect.sSize.y);
	if( fLength>20.0f || fWidth>3.0f )
		return false;

	float	fMaxZ	= cObject.GetMaxZ();
	float	fMinZ	= cObject.GetMinZ();

	//Box�̏�������ʂƂ��ė��p
	m_vecFeature[BOXSIZE_LNG]	= fLength;
	m_vecFeature[BOXSIZE_NRRW]	= fWidth;
	m_vecFeature[BOXAREA_SD]	= fLength	* fDltHeight;
	m_vecFeature[BOXAREA_FRNT]	= fWidth	* fDltHeight;
	m_vecFeature[BOXAREA_TP]	= fLength	* fWidth;
	m_vecFeature[MINBOXVOL]		= fLength	* fWidth * fDltHeight;

	//�|�C���g�N���E�h�̍��W��]
	vector<float>	vecLngPnt(pnt_size);
	vector<float>	vecNrrwPnt(pnt_size);
	vector<float>	vecHghtPnt(pnt_size);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecLngPnt[nIndex]	= (vecPntCld[nIndex].x - sTrckPose.x) * cos(-sTrckPose.psi) - (vecPntCld[nIndex].y - sTrckPose.y) * sin(-sTrckPose.psi);
		vecNrrwPnt[nIndex]	= (vecPntCld[nIndex].x - sTrckPose.x) * sin(-sTrckPose.psi) + (vecPntCld[nIndex].y - sTrckPose.y) * cos(-sTrckPose.psi);
		vecHghtPnt[nIndex]	= vecPntCld[nIndex].z;
	}
	SetParam(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	m_obMax.z = fMaxZ;
	m_obMin.z = fMinZ;
	
	//�傫���Ɋւ������
	CompSizeFtr();
	//���Ԃ���̋���
	EstimateDist(vecPntCld);
	//�ُ�ȃf�[�^���Ȃ�
	if(!CheckSgm(nLabel))
		return false;
	//���ˋ��x�̕��ρC���U
	AnalyzeIntensity(vecIntValue);
	//�`��Ɋւ�������ʁi�ʕ��3�ʁj
	AnalyzeConvexHull(cObject, vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//Box��񂩂��ʓʕ�̂̂ݗ��p
//	AnalyzeConvexHull(cObject);
	//�`��Ɋւ�������ʁi�ȉ~�t�B�b�e�B���O3�ʁj
//	CompRectFtr(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//2�l�摜�ɂ�����ʐ�
	AnalyzeBinImg(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//3���������U�s��C3�����������[�����g�s��
	CompCovMtrx(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//�|�C���g�N���E�h���z
	AnalyzePntHist(vecLngPnt, vecNrrwPnt);
	//���ˋ��x���z
	AnalyzeIntHist(vecIntValue);
	//�X���C�X������
	AnalyzeSlice(vecLngPnt, vecNrrwPnt, vecHghtPnt, vecIntValue);
	//�|�����킹
	CompMultiFtr();

	return true;
}

inline bool CSegment::SgmAnalyze(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntValue, pose_t& Mpose, int nLabel)
{
	size_t	pnt_size = vecPntCld.size();
	m_vecFeature[POINTNUM] = (float)pnt_size;
	
	// ���̂̉��
	CObjAnalyze	cObject;
	if( !cObject.Analyze(vecPntCld) )
		return false;

	// ���������镨�̂̍폜(�m�C�Y�����Ɍ���C�傫�����͎̂c���Ă����Ȃ��Ƃ��Ƃœʕ�̂ɓ����Ă��镨�̂��폜�ł��Ȃ��Ȃ�)
	RC_RECT	sRect	= cObject.GetMinAreaRect();
	float	fLength	= max(sRect.sSize.x, sRect.sSize.y);	
	if( fLength<0.2 )						
		return false;

	float	fDist		= hypot(sRect.sCenter.x, sRect.sCenter.y);
	float	fDltHeight	= cObject.GetMaxZ() - cObject.GetMinZ();
	if( fDltHeight<0.5 && fDist<30 )
		return false;

	// �傫�����镨��
	float	fWidth	= min(sRect.sSize.x, sRect.sSize.y);
	if( fLength>20.0f || fWidth>3.0f )
		return false;

	float	fMaxZ	= cObject.GetMaxZ();
	float	fMinZ	= cObject.GetMinZ();

	//Box�̏�������ʂƂ��ė��p
	m_vecFeature[BOXSIZE_LNG]	= fLength;
	m_vecFeature[BOXSIZE_NRRW]	= fWidth;
	m_vecFeature[BOXAREA_SD]	= fLength	* fDltHeight;
	m_vecFeature[BOXAREA_FRNT]	= fWidth	* fDltHeight;
	m_vecFeature[BOXAREA_TP]	= fLength	* fWidth;
	m_vecFeature[MINBOXVOL]		= fLength	* fWidth * fDltHeight;

	//�|�C���g�N���E�h�̍��W��]
	vector<float>	vecLngPnt(pnt_size);
	vector<float>	vecNrrwPnt(pnt_size);
	vector<float>	vecHghtPnt(pnt_size);
	POSE_FLT		sCenter	= sRect.sCenter;
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecLngPnt[nIndex]	= (vecPntCld[nIndex].x - sCenter.x) * cos(-sCenter.psi) - (vecPntCld[nIndex].y - sCenter.y) * sin(-sCenter.psi);
		vecNrrwPnt[nIndex]	= (vecPntCld[nIndex].x - sCenter.x) * sin(-sCenter.psi) + (vecPntCld[nIndex].y - sCenter.y) * cos(-sCenter.psi);
		vecHghtPnt[nIndex]	= vecPntCld[nIndex].z;
	}
	AnalyzeBoxInf(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	m_obMax.z = fMaxZ;
	m_obMin.z = fMinZ;
	
	//�傫���Ɋւ������
	CompSizeFtr();
	//���Ԃ���̋���
	EstimateDist(vecPntCld);
	//�ُ�ȃf�[�^���Ȃ�
	if(!CheckSgm(nLabel))
		return false;
	//���ˋ��x�̕��ρC���U
	AnalyzeIntensity(vecIntValue);
	//�`��Ɋւ�������ʁi�ʕ��3�ʁj
	AnalyzeConvexHull(cObject, vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//Box��񂩂��ʓʕ�̂̂ݗ��p
//	AnalyzeConvexHull(cObject);
	//�`��Ɋւ�������ʁi�ȉ~�t�B�b�e�B���O3�ʁj
//	CompRectFtr(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//2�l�摜�ɂ�����ʐ�
	AnalyzeBinImg(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//3���������U�s��C3�����������[�����g�s��
	CompCovMtrx(vecLngPnt, vecNrrwPnt, vecHghtPnt);
	//�|�C���g�N���E�h���z
	AnalyzePntHist(vecLngPnt, vecNrrwPnt);
	//���ˋ��x���z
	AnalyzeIntHist(vecIntValue);
	//�X���C�X������
	AnalyzeSlice(vecLngPnt, vecNrrwPnt, vecHghtPnt, vecIntValue);
	//�|�����킹
	CompMultiFtr();

	return true;
}

inline	void	CSegment::SetParam(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
{
	UINT pnt_size = vecLngPnt.size();

	//X,Y�̍ő�E�ŏ��l
	POS_FLT	obMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT obMin = CPosFlt(FLT_MAX, FLT_MAX);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		if (vecLngPnt[nIndex] > obMax.x)
			obMax.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] > obMax.y)
			obMax.y = vecNrrwPnt[nIndex];
		if (vecLngPnt[nIndex] < obMin.x)
			obMin.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] < obMin.y)
			obMin.y = vecNrrwPnt[nIndex];
	}

	//�_�Q�̏d�S
	POS3D_FLT sPosGrav;
	sPosGrav.x = sPosGrav.y = sPosGrav.z = 0.0f;
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		sPosGrav.x += vecLngPnt[nIndex];
		sPosGrav.y += vecNrrwPnt[nIndex];
		sPosGrav.z += vecHghtPnt[nIndex];
	}
	sPosGrav.x	/= (float)pnt_size;
	sPosGrav.y	/= (float)pnt_size;
	sPosGrav.z	/= (float)pnt_size;

	//�ۑ�
	m_obMax.x	= obMax.x;
	m_obMax.y	= obMax.y;
	m_obMin.x	= obMin.x;
	m_obMin.y	= obMin.y;
	m_obGrav	= sPosGrav;
}

inline void CSegment::AnalyzeBoxInf(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
{
	UINT pnt_size = vecLngPnt.size();
	
	//�ő�E�ŏ��l
	POS_FLT obMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT obMin = CPosFlt(FLT_MAX, FLT_MAX);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		if (vecLngPnt[nIndex] > obMax.x)
			obMax.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] > obMax.y)
			obMax.y = vecNrrwPnt[nIndex];
		if (vecLngPnt[nIndex] < obMin.x)
			obMin.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] < obMin.y)
			obMin.y = vecNrrwPnt[nIndex];
	}
	//X�����ӁCY���Z�ӂƂ���
	if((obMax.x - obMin.x) < (obMax.y - obMin.y)){
		vecLngPnt.swap(vecNrrwPnt);
		swap(obMax.x, obMax.y);
		swap(obMin.x, obMin.y);
	}

	//�_�Q�̏d�S
	POS3D_FLT sPosGrav;
	sPosGrav.x = sPosGrav.y = sPosGrav.z = 0.0f;
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		sPosGrav.x += vecLngPnt[nIndex];
		sPosGrav.y += vecNrrwPnt[nIndex];
		sPosGrav.z += vecHghtPnt[nIndex];
	}
	sPosGrav.x	/= (float)pnt_size;
	sPosGrav.y	/= (float)pnt_size;
	sPosGrav.z	/= (float)pnt_size;

	//���o�^
	m_obMax.x = obMax.x;
	m_obMax.y = obMax.y;
	m_obMin.x = obMin.x;
	m_obMin.y = obMin.y;
	m_obGrav	= sPosGrav;
}

inline void CSegment::CompSizeFtr()
{
	//��
	float fLongSide		= m_obMax.x - m_obMin.x;
	float fNarrowSide	= m_obMax.y - m_obMin.y;
	//����
	float fHeight		= m_obMax.z - m_obMin.z;

	m_vecFeature[LONGSIDE]		= fLongSide;
	m_vecFeature[NARROWSIDE]	= fNarrowSide;
	m_vecFeature[HEIGHT]		= fHeight;

	//�ő�E�ŏ�����
	m_vecFeature[MAXHEIGHT]		= m_obMax.z;
	m_vecFeature[MINHEIGHT]		= m_obMin.z;
}

inline void CSegment::AnalyzeIntensity(vector<float>& vecIntValue)
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
		// ���ˋ��x���L���Ȃ��̂������g�p
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
		// ���ϒl
		fAvgInt = fAvgInt/(float)count;
		// ���U�l
		fVarInt = fSqInt/(float)count - fAvgInt*fAvgInt;
	}	

	m_vecFeature[INTENSITY]		= fAvgInt;
	m_vecFeature[VARINTENSITY]	= fVarInt;

	//�ő�E�ŏ��l
	m_vecFeature[INTMAX]		= fMaxInt;
	m_vecFeature[INTMIN]		= fMinInt;
}

inline void CSegment::EstimateDist(vector<POS3D_FLT>& vecPntCld, pose_t& Mpose)
{
	size_t	stPntSize	= vecPntCld.size();
	POS_FLT sPosGrav = CPosFlt(0,0);

	for (int nIndex = 0; nIndex < stPntSize; nIndex++)
	{
		sPosGrav.x += vecPntCld[nIndex].x;
		sPosGrav.y += vecPntCld[nIndex].y;
	}
	//X,Y�d�S
	sPosGrav.x = sPosGrav.x / (float)stPntSize;
	sPosGrav.y = sPosGrav.y / (float)stPntSize;

	float fDist = hypot( sPosGrav.x-(float)Mpose.x, sPosGrav.y-(float)Mpose.y );

	m_vecFeature[DISTANCE] = fDist;
}

inline void CSegment::EstimateDist(vector<POS3D_FLT>& vecPntCld)
{
	size_t	stPntSize	= vecPntCld.size();
	POS_FLT sPosGrav	= CPosFlt(0,0);

	for (int nIndex = 0; nIndex < stPntSize; nIndex++)
	{
		sPosGrav.x += vecPntCld[nIndex].x;
		sPosGrav.y += vecPntCld[nIndex].y;
	}
	// X, Y�d�S
	sPosGrav.x = sPosGrav.x / (float)stPntSize;
	sPosGrav.y = sPosGrav.y / (float)stPntSize;

	float fDist = hypot(sPosGrav.x, sPosGrav.y);

	m_vecFeature[DISTANCE] = fDist;
}

inline void CSegment::AnalyzeConvexHull(CObjAnalyze& cObj, vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
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
		vecTpConvexHull.push_back( CObjAnalyze::Conv(*itrContour) );

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

inline void CSegment::AnalyzeConvexHull(CObjAnalyze& cObj)
{
	//��ʂ�Box���𗘗p
	vector<Point2f>		vecTpConvexHull;
	vector<POS_FLT>&	vecContour = cObj.GetContour();
	vecTpConvexHull.reserve(vecContour.size());
	vector<POS_FLT>::iterator	itrContour;
	for( itrContour=vecContour.begin(); itrContour!=vecContour.end(); itrContour++ )
		vecTpConvexHull.push_back( CObjAnalyze::Conv(*itrContour) );

	//�ʐ�
	double TpArea		= contourArea(vecTpConvexHull);
	//���͒���
	double TpLength		= arcLength(vecTpConvexHull,true);

	m_vecFeature[TOPAREA]		= (float)TpArea;
	m_vecFeature[TOPPER]		= (float)TpLength;
}

inline void CSegment::CompRectFtr(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
{
	UINT	pnt_size = vecLngPnt.size();

	//OpennCV�p�Ɍ^�ϊ�
	vector<Point2f>	vecSdCvPnts(pnt_size);
	vector<Point2f>	vecFrntCvPnts(pnt_size);
	vector<Point2f>	vecTpCvPnts(pnt_size);
	for (UINT nLoop = 0; nLoop < pnt_size; nLoop++)
	{
		vecSdCvPnts[nLoop]		= Point2f(vecLngPnt[nLoop], vecHghtPnt[nLoop]);
		vecFrntCvPnts[nLoop]	= Point2f(vecNrrwPnt[nLoop], vecHghtPnt[nLoop]);
		vecTpCvPnts[nLoop]		= Point2f(vecLngPnt[nLoop], vecNrrwPnt[nLoop]);
	}

	//�ȉ~�t�B�b�e�B���O
	AnalyzeEllipse(vecSdCvPnts, vecFrntCvPnts, vecTpCvPnts);
}

inline void CSegment::AnalyzeEllipse(vector<Point2f>& vecSdCvPnts, vector<Point2f>& vecFrntCvPnts, vector<Point2f>& vecTpCvPnts)
{
	//�ȉ~�t�B�b�e�B���O
	RotatedRect	sSd_RotRect		= fitEllipse(vecSdCvPnts);
	RotatedRect	sFrnt_RotRect	= fitEllipse(vecFrntCvPnts);
	RotatedRect	sTp_RotRect		= fitEllipse(vecTpCvPnts);

	float fSdWidth, fSdHeight, fSdAngle, fFrntWidth, fFrntHeight, fFrntAngle, fTpWidth, fTpHeight, fTpAngle;
	//����
	fSdWidth	= sSd_RotRect.size.width;
	fSdHeight	= sSd_RotRect.size.height;
	fSdAngle	= sSd_RotRect.angle;
	//����
	fFrntWidth	= sFrnt_RotRect.size.width;
	fFrntHeight = sFrnt_RotRect.size.height;
	fFrntAngle	= sFrnt_RotRect.angle;
	//���
	fTpWidth	= sTp_RotRect.size.width;
	fTpHeight	= sTp_RotRect.size.height;
	fTpAngle	= sTp_RotRect.angle;

	//0-179deg�ɕϊ�
	if(fSdAngle >= 180.0f)
			fSdAngle -= 180.0f;
	if(fFrntAngle >= 180.0f)
			fFrntAngle -= 180.0f;
	if(fTpAngle >= 180.0f)
			fTpAngle -= 180.0f;

	//45deg-135deg��height->width
	if(45.0f < fSdAngle && fSdAngle < 135.0f)
		swap(fSdWidth, fSdHeight);
	if(45.0f < fFrntAngle && fFrntAngle < 135.0f)
		swap(fFrntWidth, fFrntHeight);
	if(45.0f < fTpAngle && fTpAngle < 135.0f)
		swap(fTpWidth, fTpHeight);

	m_vecFeature[SD_ELLPSLNG]		= fSdHeight;
	m_vecFeature[SD_ELLPSNRRW]		= fSdWidth;
	m_vecFeature[SD_ELLPSANGL]		= fSdAngle;
	m_vecFeature[FRNT_ELLPSLNG]		= fFrntHeight;
	m_vecFeature[FRNT_ELLPSNRRW]	= fFrntWidth;
	m_vecFeature[FRNT_ELLPSANGL]	= fFrntAngle;
	m_vecFeature[TP_ELLPSLNG]		= fTpHeight;
	m_vecFeature[TP_ELLPSNRRW]		= fTpWidth;
	m_vecFeature[TP_ELLPSANGL]		= fTpAngle;
}

inline bool CSegment::CheckSgm(int nLabel)
{
//	float fMaxSize		= MAX_SIZE[nLabel];
	float fMaxSize		= FLTMAX_SIZE[nLabel];
//	float fMaxHeight	= MAX_HEIGHT[nLabel];
	float fMaxHeight	= FLTMAX_SIZE[nLabel];

	//���ӁC�������傫������Ζ���
	if(m_vecFeature[LONGSIDE] > fMaxSize || m_vecFeature[HEIGHT] > fMaxHeight)
		return false;

	//�w��|�C���g���͈̔͊O�͖���
	if(m_vecFeature[POINTNUM] < MIN_POINT || m_vecFeature[POINTNUM] > MAX_POINT)
		return false;

	//�w�苗���O�͖���
	if(m_vecFeature[DISTANCE] < MIN_DISTANCE || m_vecFeature[DISTANCE] > MAX_DISTANCE)
		return false;
	
	return true;	
}

inline void CSegment::CompMultiFtr()
{
	float epsilon_three = 0.001f;
	float epsilon_two	= 0.01f;
	float fLngSd, fNrrwSd, fHght, fPntNm, fInt, fVar, fSdA, fFrntA, fTpA, fSdPr, fFrntPr, fTpPr, fSdEllpsL, fSdEllpsN, fFrntEllpsL, fFrntEllpsN, fTpEllpsL, fTpEllpsN, fSdNonZ, fFrntNonZ, fTpNonZ;
	
	fLngSd	= m_vecFeature[LONGSIDE];
	fNrrwSd = m_vecFeature[NARROWSIDE];
	fHght	= m_vecFeature[HEIGHT];
	fPntNm	= m_vecFeature[POINTNUM];
	fInt	= m_vecFeature[INTENSITY];
	fVar	= m_vecFeature[VARINTENSITY];
	fSdA	= m_vecFeature[SIDEAREA];
	fFrntA	= m_vecFeature[FRONTAREA];
	fTpA	= m_vecFeature[TOPAREA];
	fSdPr	= m_vecFeature[SIDEPER];
	fFrntPr = m_vecFeature[FRONTPER];
	fTpPr	= m_vecFeature[TOPPER];

	fSdEllpsL	= m_vecFeature[SD_ELLPSLNG];
	fSdEllpsN	= m_vecFeature[SD_ELLPSNRRW];
	fFrntEllpsL = m_vecFeature[FRNT_ELLPSLNG];
	fFrntEllpsN = m_vecFeature[FRNT_ELLPSNRRW];
	fTpEllpsL	= m_vecFeature[TP_ELLPSLNG];
	fTpEllpsN	= m_vecFeature[TP_ELLPSNRRW];

	fSdNonZ		= m_vecFeature[SD_NONZERO];
	fFrntNonZ	= m_vecFeature[FRNT_NONZERO];
	fTpNonZ		= m_vecFeature[TP_NONZERO];

	m_vecFeature[VOLBOX]		= fLngSd * fNrrwSd * fHght;
	m_vecFeature[MUL_LNGNRRW]	= fLngSd * fNrrwSd;
	m_vecFeature[MUL_LNGHGHT]	= fLngSd * fHght;

	m_vecFeature[DIV_VARINT]	= fVar	 / (fInt				+ epsilon_two);
	m_vecFeature[PNTDENSITY]	= fPntNm / (fLngSd	* fHght		+ epsilon_two);
	m_vecFeature[DIV_HGHTLNG]	= fHght	 / (fLngSd				+ epsilon_three);
	m_vecFeature[DIV_HGHTNRRW]	= fHght	 / (fNrrwSd				+ epsilon_three);

	m_vecFeature[SDDENS]		= fSdA	 / (fLngSd	* fHght		+ epsilon_two);
	m_vecFeature[FRNTDENS]		= fFrntA / (fNrrwSd	* fHght		+ epsilon_two);
	m_vecFeature[TPDENS]		= fTpA	 / (fLngSd	* fNrrwSd	+ epsilon_two);
	m_vecFeature[SDCIRCLE]		= fSdA	 / (fSdPr	* fSdPr		+ epsilon_two);
	m_vecFeature[FRNTCIRCLE]	= fFrntA / (fFrntPr	* fFrntPr	+ epsilon_two);
	m_vecFeature[TPCIRCLE]		= fTpA	 / (fTpPr	* fTpPr		+ epsilon_two);
	m_vecFeature[SDELONG]		= fSdA	 / (fLngSd				+ epsilon_three);
	m_vecFeature[FRNTELONG]		= fFrntA / (fNrrwSd				+ epsilon_three);
	m_vecFeature[TPELONG]		= fTpA	 / (fLngSd				+ epsilon_three);

	m_vecFeature[SDDENS_NONZ]		= fSdNonZ	/ (fSdA		+ epsilon_three);
	m_vecFeature[FRNTDENS_NONZ]		= fFrntNonZ / (fFrntA	+ epsilon_three);
	m_vecFeature[TPDENS_NONZ]		= fTpNonZ	/ (fTpA		+ epsilon_three);
	m_vecFeature[SDELONG_NONZ]		= fSdNonZ	/ (fLngSd	+ epsilon_three);
	m_vecFeature[FRNTELONG_NONZ]	= fFrntNonZ / (fNrrwSd	+ epsilon_three);
	m_vecFeature[TPELONG_NONZ]		= fTpNonZ	/ (fLngSd	+ epsilon_three);
}

inline void CSegment::AnalyzeBinImg(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
{
	UINT pnt_size = vecLngPnt.size();

	//X,Y,Z�̒��S���W
	POS3D_FLT obCenter;
	obCenter.x = 0.5*(m_obMax.x + m_obMin.x);
	obCenter.y = 0.5*(m_obMax.y + m_obMin.y);
	obCenter.z = 0.5*(m_obMax.z	+ m_obMin.z);
	//�E�B���h�E�T�C�Y(float)
	POS3D_FLT fWindSize;
	fWindSize.x = (m_obMax.x-m_obMin.x) / PIXEL_SIZE + 10.0f;
	fWindSize.y = (m_obMax.y-m_obMin.y) / PIXEL_SIZE + 10.0f;
	fWindSize.z = (m_obMax.z-m_obMin.z)	/ PIXEL_SIZE + 10.0f;
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

inline void CSegment::CompCovMtrx(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt)
{
	UINT		pnt_size = vecLngPnt.size();
	POS3D_FLT	sPosGrav = GetPntGrav();

	POS3D_FLT sPosDlt;
	vector<float> vecCovElmnt(COVMTRX_ELNUM, 0.0f);		//3���������U�s��
	vector<float> vecInrElmnt(INRMTRX_ELNUM, 0.0f);		//3�����������[�����g�s��
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		//�d�S�𒆐S�Ƃ������W
		sPosDlt.x = vecLngPnt[nIndex]	- sPosGrav.x;
		sPosDlt.y = vecNrrwPnt[nIndex]	- sPosGrav.y;
		sPosDlt.z = vecHghtPnt[nIndex]	- sPosGrav.z;
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

inline void CSegment::AnalyzePntHist(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt)
{
	UINT pnt_size = vecLngPnt.size();

	float fLngGrav	= m_obGrav.x;
	float fNrrwGrav = m_obGrav.y;
	
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

inline void CSegment::CompPntHist(vector<int>& vecPntHist, vector<float>& vecPnt, float fGrav, float fBlckSize, float fWdthRng)
{
	UINT		pnt_size	= vecPnt.size();

	int			nLoopNum	= int(0.5 + fWdthRng/fBlckSize);	//��������������(�l�̌ܓ�)
	int			tmp			= int(0.5 * nLoopNum);
	//�����_�ŏ��l
	float		fMinHrz		= fGrav - 0.5*fBlckSize - fBlckSize*(float)tmp;		

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

inline void CSegment::AnalyzeIntHist(vector<float>& vecIntValue)
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

inline void CSegment::AnalyzeSlice(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, vector<float>& vecIntValue)
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

inline void CSegment::CompLayer(vector<float>& vecHghtPnt, vector<int>& vecPartNum)
{
	UINT		pnt_size = vecHghtPnt.size();
	vector<int> vecLayerNum(pnt_size);

	UINT	nLoopNum	= LAYERNUM;						//���C���[�̐�
	float	fWdith		= SLICE_MAXHGHT-SLICE_MINHGHT;	//�S��
	float	fMinHght	= SLICE_MINHGHT;				//�����̍ŏ��l
	
	int		num;
	float	fDlt;
	float	fRate;
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		//�ŏ��l�Ƃ̍�
		fDlt	= vecHghtPnt[nIndex] - fMinHght;
		//�S���ɑ΂��銄��0-1
		fRate	= fDlt / fWdith;
		//���C���[�̔ԍ�
		num = int(fRate * nLoopNum);
		if(num < 0)
			num = 0;
		else if(num > nLoopNum-1)
			num = nLoopNum -1;

		vecLayerNum[nIndex] = num;
	}

	vecPartNum.swap(vecLayerNum);
}

inline void	CSegment::CompPartWdth(vector<float>& vecPntCld, vector<float>& vecPartWdth, vector<int>& vecPartNum)
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

inline void CSegment::AnalyzePartInt(vector<float>& vecIntValue, vector<float>& vecAvrgInt, vector<int>& vecPartNum)
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

inline void CSegment::ShowSgmImg(vector<POS3D_FLT>& vecPntCld, vector<float>& vecIntensity, float pixel_size)
{/*
	size_t pnt_size = vecPntCld.size();

	//BoundingBox�쐬->���̊p�x�擾
	CBoxAnalyze			cBox;
	cBox.SetPointCloud(vecPntCld);
	cBox.EstimateSub();

	//���擾
	POSE_FLT sCntr = cBox.GetCenter();
	float	 fMaxZ = cBox.GetMaxZ(); 
	float	 fMinZ = cBox.GetMinZ();
	//���W��]
	vector<POS_FLT> vecRotPntCld(pnt_size);
	//�ő�ŏ��l
	POS_FLT obMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT obMin = CPosFlt(FLT_MAX, FLT_MAX);

	for (int nIndex = 0; nIndex < pnt_size; nIndex++){
		//	x,y���W��]
		vecRotPntCld[nIndex].x = (vecPntCld[nIndex].x - sCntr.x) * cos(-sCntr.psi) - (vecPntCld[nIndex].y - sCntr.y) * sin(-sCntr.psi);
		vecRotPntCld[nIndex].y = (vecPntCld[nIndex].x - sCntr.x) * sin(-sCntr.psi) + (vecPntCld[nIndex].y - sCntr.y) * cos(-sCntr.psi);

		if (vecRotPntCld[nIndex].x > obMax.x){
			obMax.x = vecRotPntCld[nIndex].x;
		}
		if (vecRotPntCld[nIndex].y > obMax.y){
			obMax.y = vecRotPntCld[nIndex].y;
		}
		if (vecRotPntCld[nIndex].x < obMin.x){
			obMin.x = vecRotPntCld[nIndex].x;
		}
		if (vecRotPntCld[nIndex].y < obMin.y){
			obMin.y = vecRotPntCld[nIndex].y;
		}
	}

	//x, y, z�̒��S���W
	POS3D_FLT obCenter;
	obCenter.x = 0.5f*(obMax.x + obMin.x);
	obCenter.y = 0.5f*(obMax.y + obMin.y);
	obCenter.z = 0.5f*(fMaxZ + fMinZ);
	//�E�B���h�E�T�C�Y(float)
	POS3D_FLT WindowSize_FLT;
	WindowSize_FLT.x = (obMax.x - obMin.x)	/ pixel_size + 100.0f;
	WindowSize_FLT.y = (obMax.y - obMin.y)	/ pixel_size + 100.0f;
	WindowSize_FLT.z = (fMaxZ - fMinZ)		/ pixel_size + 100.0f;
	//�摜x, y, z���S
	POS3D_FLT pxCenter;
	pxCenter.x = 0.5f*WindowSize_FLT.x;
	pxCenter.y = 0.5f*WindowSize_FLT.y;
	pxCenter.z = 0.5f*WindowSize_FLT.z;
	//�؂�̂�
	int WindowSizeX = floor(WindowSize_FLT.x);
	int WindowSizeY = floor(WindowSize_FLT.y);
	int WindowSizeZ = floor(WindowSize_FLT.z);
	//3�����摜����
	cv::Mat Side_img	= cv::Mat::zeros(cv::Size(WindowSizeX, WindowSizeZ), CV_8UC1);
	cv::Mat Front_img	= cv::Mat::zeros(cv::Size(WindowSizeY, WindowSizeZ), CV_8UC1);
	cv::Mat Top_img		= cv::Mat::zeros(cv::Size(WindowSizeX, WindowSizeY), CV_8UC1); 

	POS3D_FLT delta;
	int PixelX;
	int PixelY1;
	int PixelY2;
	int PixelZ;

	for (int i = 0; i < pnt_size; ++i){
		//���S���W����̍�
		delta.x = (vecRotPntCld[i].x - obCenter.x)	/ (float)pixel_size;
		delta.y = (vecRotPntCld[i].y - obCenter.y)	/ (float)pixel_size;
		delta.z = (vecPntCld[i].z - obCenter.z)		/ (float)pixel_size;
		//�摜���W
		PixelX	= int(pxCenter.x + delta.x);
		PixelY1 = int(pxCenter.y + delta.y);
		PixelY2 = int(pxCenter.y - delta.y);
		PixelZ	= int(pxCenter.z - delta.z);
		//�P�x�t�^
		Side_img.at<uchar>(PixelZ, PixelX)		= 255 * vecIntensity[i];
		Front_img.at<uchar>(PixelZ, PixelY1)	= 255 * vecIntensity[i];
		Top_img.at<uchar>(PixelY2, PixelX)		= 255 * vecIntensity[i];
	}

	//��ʉ摜�̌����𑵂���(����)
	if (WindowSizeY > WindowSizeX){
		cv::Mat tmp_img = Top_img;
		cv::resize(Top_img, Top_img, cv::Size(WindowSizeY, WindowSizeX));
		//��](�A�t�B���ϊ�)
		const cv::Point2f src_pt[] = { cv::Point2f(0, 0), cv::Point2f(WindowSizeX, 0), cv::Point2f(0, WindowSizeY) };
		const cv::Point2f dst_pt[] = { cv::Point2f(WindowSizeY, 0), cv::Point2f(WindowSizeY, WindowSizeX), cv::Point2f(0, 0) };
		const cv::Mat affine_matrix = cv::getAffineTransform(src_pt, dst_pt);
		cv::warpAffine(tmp_img, Top_img, affine_matrix, Top_img.size());
		//���ʂƑ��ʂ����ւ�
		cv::swap(Side_img, Front_img);
	}

	//�����t�H���W�[���Z
	cv::Mat element(2, 2, CV_8U, cv::Scalar::all(255));
	cv::morphologyEx(Side_img, Side_img, cv::MORPH_DILATE, element, cv::Point(-1, -1), 1);
	cv::morphologyEx(Front_img, Front_img, cv::MORPH_DILATE, element, cv::Point(-1, -1), 1);
	cv::morphologyEx(Top_img, Top_img, cv::MORPH_DILATE, element, cv::Point(-1, -1), 1);

	cv::namedWindow("Side Image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::namedWindow("Front Image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::namedWindow("Top Image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("Side Image", Side_img);
	cv::imshow("Front Image", Front_img);
	cv::imshow("Top Image", Top_img);

	return;*/
}

inline void CSegment::ShowSgmImgColor(vector<POS3D_FLT>& vecPntCld, POSE_FLT sPose, float pixel_size)
{
	//�|�C���g��
	UINT	pnt_size = vecPntCld.size();

	// ���̂̉��
	CObjAnalyze	cObject;
	if( !cObject.Analyze(vecPntCld) )
		return;

	//���擾
	float		fMaxZ	= cObject.GetMaxZ();
	float		fMinZ	= cObject.GetMinZ();

	//�|�C���g�N���E�h�̍��W��]
	vector<float>	vecLngPnt(pnt_size);
	vector<float>	vecNrrwPnt(pnt_size);
	vector<float>	vecHghtPnt(pnt_size);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecLngPnt[nIndex]	= (vecPntCld[nIndex].x - sPose.x) * cos(-sPose.psi) - (vecPntCld[nIndex].y - sPose.y) * sin(-sPose.psi);
		vecNrrwPnt[nIndex]	= (vecPntCld[nIndex].x - sPose.x) * sin(-sPose.psi) + (vecPntCld[nIndex].y - sPose.y) * cos(-sPose.psi);
		vecHghtPnt[nIndex]	= vecPntCld[nIndex].z;
	}

	//�ő�E�ŏ��l
	POS_FLT obMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT obMin = CPosFlt(FLT_MAX, FLT_MAX);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		if (vecLngPnt[nIndex] > obMax.x)
			obMax.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] > obMax.y)
			obMax.y = vecNrrwPnt[nIndex];
		if (vecLngPnt[nIndex] < obMin.x)
			obMin.x = vecLngPnt[nIndex];
		if (vecNrrwPnt[nIndex] < obMin.y)
			obMin.y = vecNrrwPnt[nIndex];
	}

	//x, y, z�̒��S���W
	POS3D_FLT obCenter;
	obCenter.x = 0.5f*(obMax.x	+ obMin.x);
	obCenter.y = 0.5f*(obMax.y	+ obMin.y);
	obCenter.z = 0.5f*(fMaxZ	+ fMinZ);
	//�E�B���h�E�T�C�Y(float)
	POS3D_FLT WindowSize_FLT;
	WindowSize_FLT.x = (obMax.x - obMin.x)	/ pixel_size + 300.0f;
	WindowSize_FLT.y = (obMax.y - obMin.y)	/ pixel_size + 300.0f;
	WindowSize_FLT.z = (fMaxZ	- fMinZ)	/ pixel_size + 300.0f;
	//�摜x, y, z���S
	POS3D_FLT pxCenter;
	pxCenter.x = 0.5f*WindowSize_FLT.x;
	pxCenter.y = 0.5f*WindowSize_FLT.y;
	pxCenter.z = 0.5f*WindowSize_FLT.z;
	//�؂�̂�
	int WindowSizeX = floor(WindowSize_FLT.x);
	int WindowSizeY = floor(WindowSize_FLT.y);
	int WindowSizeZ = floor(WindowSize_FLT.z);

	//3�����摜����
	cv::Size	img_sizeSD(WindowSizeX, WindowSizeZ);
	cv::Size	img_sizeFRNT(WindowSizeY, WindowSizeZ);
	cv::Size	img_sizeTP(WindowSizeX, WindowSizeY);
	cv::Mat		imgSD(img_sizeSD, CV_8UC3, cv::Scalar::all(255));
	cv::Mat		imgFRNT(img_sizeFRNT, CV_8UC3, cv::Scalar::all(255));
	cv::Mat		imgTP(img_sizeTP, CV_8UC3, cv::Scalar::all(255));

	POS3D_FLT delta;
	int PixelX;
	int PixelY1;
	int PixelY2;
	int PixelZ;
	vector<Point>	vecSdCvPnts(pnt_size);
	vector<Point>	vecFrntCvPnts(pnt_size);
	vector<Point>	vecTpCvPnts(pnt_size);
	for (int i = 0; i < pnt_size; ++i){
		//���S���W����̍�
		delta.x = (vecLngPnt[i]		- obCenter.x)	/ (float)pixel_size;
		delta.y = (vecNrrwPnt[i]	- obCenter.y)	/ (float)pixel_size;
		delta.z = (vecHghtPnt[i]	- obCenter.z)	/ (float)pixel_size;
		//�摜���W
		PixelX	= int(pxCenter.x + delta.x);
		PixelY1 = int(pxCenter.y + delta.y);
		PixelY2 = int(pxCenter.y - delta.y);
		PixelZ	= int(pxCenter.z - delta.z);
		vecSdCvPnts[i]		= Point2f(PixelX, PixelZ);	// OpenCV�p�Ɍ^�ϊ�
		vecFrntCvPnts[i]	= Point2f(PixelY1, PixelZ);	// OpenCV�p�Ɍ^�ϊ�
		vecTpCvPnts[i]		= Point2f(PixelX, PixelY2);	// OpenCV�p�Ɍ^�ϊ�
		////�P�x�t�^
		//imgSD.at<Vec3b>(PixelZ, PixelX) = (255, 0, 0);
		//imgFRNT.at<Vec3b>(PixelZ, PixelY1) = (255, 0, 0);
		//imgTP.at<Vec3b>(PixelY2, PixelX) = (255, 0, 0);
		//�P�x�t�^
		//cv::circle(imgSD, cv::Point(PixelX, PixelZ), 8, cv::Scalar(62,116,80), -1, CV_AA);
		//cv::circle(imgFRNT, cv::Point(PixelY1, PixelZ), 8, cv::Scalar(62,116,80), -1, CV_AA);
		//cv::circle(imgTP, cv::Point(PixelX, PixelY2), 8, cv::Scalar(62,116,80), -1, CV_AA);
		//cv::circle(imgSD, cv::Point(PixelX, PixelZ), 8, cv::Scalar(0,0,200), -1, CV_AA);
		//cv::circle(imgFRNT, cv::Point(PixelY1, PixelZ), 8, cv::Scalar(0,0,200), -1, CV_AA);
		//cv::circle(imgTP, cv::Point(PixelX, PixelY2), 8, cv::Scalar(0,0,200), -1, CV_AA);
		cv::circle(imgSD, cv::Point(PixelX, PixelZ), 8, cv::Scalar(178,25,39), -1, CV_AA);
		cv::circle(imgFRNT, cv::Point(PixelY1, PixelZ), 8, cv::Scalar(178,25,39), -1, CV_AA);
		cv::circle(imgTP, cv::Point(PixelX, PixelY2), 8, cv::Scalar(178,25,39), -1, CV_AA);
	}
	////��ʉ摜�̌����𑵂���(����)
	//if (WindowSizeY > WindowSizeX){
	//	cv::Mat tmp_img = imgTP;
	//	cv::resize(imgTP, imgTP, cv::Size(WindowSizeY, WindowSizeX));
	//	//��](�A�t�B���ϊ�)
	//	const cv::Point2f src_pt[] = { cv::Point2f(0, 0), cv::Point2f(WindowSizeX, 0), cv::Point2f(0, WindowSizeY) };
	//	const cv::Point2f dst_pt[] = { cv::Point2f(WindowSizeY, 0), cv::Point2f(WindowSizeY, WindowSizeX), cv::Point2f(0, 0) };
	//	const cv::Mat affine_matrix = cv::getAffineTransform(src_pt, dst_pt);
	//	cv::warpAffine(tmp_img, imgTP, affine_matrix, imgTP.size());
	//	//���ʂƑ��ʂ����ւ�
	//	cv::swap(imgSD, imgFRNT);
	//}

//	RotatedRect	sSd_RotRect		= fitEllipse(vecSdCvPnts);
//	RotatedRect	sFrnt_RotRect	= fitEllipse(vecFrntCvPnts);
//	RotatedRect	sTp_RotRect		= fitEllipse(vecTpCvPnts);
//	ellipse(imgSD, sSd_RotRect, cv::Scalar(0,0,255), 2, CV_AA);
//	ellipse(imgFRNT, sFrnt_RotRect, cv::Scalar(0,0,255), 2, CV_AA);
//	ellipse(imgTP, sTp_RotRect, cv::Scalar(0,0,255), 2, CV_AA);
	//cout << "<ImageEllipse>" << " Side " << sSd_RotRect.size.width << " " << sSd_RotRect.size.height << " Front " << sFrnt_RotRect.size.width << " " << sFrnt_RotRect.size.height << " Top " << sTp_RotRect.size.width << " " << sTp_RotRect.size.height << endl;
	//cout << "angle:" << sSd_RotRect.angle << " " << sFrnt_RotRect.angle << " " << sTp_RotRect.angle << endl;

	// CV_*C2�^��Mat�ɕϊ����Ă���C�ʕ���v�Z
//	std::vector<cv::Point> hullSD;
//	std::vector<cv::Point> hullFRNT;
//	std::vector<cv::Point> hullTP;
//	cv::convexHull(vecSdCvPnts, hullSD);
//	cv::convexHull(vecFrntCvPnts, hullFRNT);
//	cv::convexHull(vecTpCvPnts, hullTP);
//	// �ʕ��`��
//	int hnumSD = hullSD.size();
//	int hnumFRNT = hullFRNT.size();
//	int hnumTP = hullTP.size();
//
//	for(int i=0; i<hnumSD; ++i)
//		cv::line(imgSD, hullSD[i], hullSD[i+1<hnumSD?i+1:0], cv::Scalar(0,0,200), 3, CV_AA);
//	for(int i=0; i<hnumFRNT; ++i)
//		cv::line(imgFRNT, hullFRNT[i], hullFRNT[i+1<hnumFRNT?i+1:0], cv::Scalar(0,0,200), 3, CV_AA);
//	for(int i=0; i<hnumTP; ++i)
//		cv::line(imgTP, hullTP[i], hullTP[i+1<hnumTP?i+1:0], cv::Scalar(0,0,200), 3, CV_AA);

//	double Lngline = 0.6;
//	for(int iLoop = -5; iLoop < 5; ++iLoop){
//		int a = pxCenter.x + Lngline*iLoop/pixel_size;
//		int b = pxCenter.y + Lngline*iLoop/pixel_size;
//		int c = pxCenter.z + Lngline*iLoop/pixel_size;
//		cv::line(imgSD, cv::Point(a, 0), cv::Point(a, WindowSizeZ), cv::Scalar(0,0,200), 3, 4);
//		cv::line(imgSD, cv::Point(0, c), cv::Point(WindowSizeX, c), cv::Scalar(0,0,200), 3, 4);
//		cv::line(imgFRNT, cv::Point(b, 0), cv::Point(b, WindowSizeZ), cv::Scalar(0,0,200), 3, 4);
//		cv::line(imgFRNT, cv::Point(0, c), cv::Point(WindowSizeY, c), cv::Scalar(0,0,200), 3, 4);
//		cv::line(imgTP, cv::Point(a, 0), cv::Point(a, WindowSizeY), cv::Scalar(0,0,200), 3, 4);
//		cv::line(imgTP, cv::Point(0, b), cv::Point(WindowSizeX, b), cv::Scalar(0,0,200), 3, 4);
//	}

	cv::namedWindow("imageSD", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	cv::namedWindow("imageFRNT", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	cv::namedWindow("imageTP", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
	cv::imshow("imageSD", imgSD);
	cv::imshow("imageFRNT", imgFRNT);
	cv::imshow("imageTP", imgTP);
}

inline void CSegment::RotPntCld(vector<POS3D_FLT>& vecPntCld, vector<POS3D_FLT>& vecRotPntCld)
{
	size_t pnt_size = vecPntCld.size();
	vecRotPntCld.resize(pnt_size);

	POS3D_FLT sOffset;
	sOffset.x = 0.91f;
	sOffset.y = 0.0f;
	sOffset.z = 2.08f;

	float fRoll		= 0.186f;
	float fPitch	= 0.835f;
	float fYaw		= 0.0f;

	float SinR		= sin(Deg2Rad(fRoll));
	float CosR		= cos(Deg2Rad(fRoll));
	float SinP		= sin(Deg2Rad(fPitch));
	float CosP		= cos(Deg2Rad(fPitch));
	float SinY		= sin(Deg2Rad(fYaw));
	float CosY		= cos(Deg2Rad(fYaw));

	float tmpX;
	float tmpY;
	float tmpZ;
	for(int nIndex = 0; nIndex < pnt_size; ++nIndex){
		tmpX = vecPntCld[nIndex].x;
		tmpY = vecPntCld[nIndex].y;
		tmpZ = vecPntCld[nIndex].z;
		vecRotPntCld[nIndex].x = tmpX*CosR*CosP + tmpY*(CosR*SinP*SinY - SinR*CosY)	+ tmpZ*(CosR*SinP*CosY + SinR*SinY)	;//	+ sOffset.x;
		vecRotPntCld[nIndex].y = tmpX*SinR*CosP + tmpY*(SinR*SinP*SinY + CosR*CosY)	+ tmpZ*(SinR*SinP*CosY - CosR*SinY)	;//	+ sOffset.y;
		vecRotPntCld[nIndex].z = tmpX*(-SinP)	+ tmpY*CosP*SinY					+ tmpZ*CosP*CosY					;//	+ sOffset.y;
	}
}

inline void CSegment::ShowFalse(vector<POS3D_FLT>& vecPntCld, POSE_FLT sPose, stringstream& sstrText, cv::Scalar& color, string& strFileName)
{
	size_t pnt_size = vecPntCld.size();

	// ���W��]
	vector<POS3D_FLT> vecRotPntCld(pnt_size);
	for(int nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecRotPntCld[nIndex].x = (vecPntCld[nIndex].x-sPose.x)*cos(-sPose.psi) - (vecPntCld[nIndex].y-sPose.y)*sin(-sPose.psi);
		vecRotPntCld[nIndex].y = (vecPntCld[nIndex].x-sPose.x)*sin(-sPose.psi) + (vecPntCld[nIndex].y-sPose.y)*cos(-sPose.psi);
		vecRotPntCld[nIndex].z = vecPntCld[nIndex].z;
	}

	// �ő�E�ŏ��l
	POS_FLT obMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT obMin = CPosFlt(FLT_MAX, FLT_MAX);
	float	fMaxZ = -FLT_MAX;
	float	fMinZ = FLT_MAX;
	for(int nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		if(vecRotPntCld[nIndex].x > obMax.x)
			obMax.x = vecRotPntCld[nIndex].x;
		if(vecRotPntCld[nIndex].y > obMax.y)
			obMax.y = vecRotPntCld[nIndex].y;
		if(vecRotPntCld[nIndex].z > fMaxZ)
			fMaxZ = vecRotPntCld[nIndex].z;
		if(vecRotPntCld[nIndex].x < obMin.x)
			obMin.x = vecRotPntCld[nIndex].x;
		if(vecRotPntCld[nIndex].y < obMin.y)
			obMin.y = vecRotPntCld[nIndex].y;
		if(vecRotPntCld[nIndex].z < fMinZ)
			fMinZ = vecRotPntCld[nIndex].z;
	}

	// ���S���W
	POS3D_FLT obCenter;
	obCenter.x = 0.5f*(obMax.x	+ obMin.x);
	obCenter.y = 0.5f*(obMax.y	+ obMin.y);
	obCenter.z = 0.5f*(fMaxZ	+ fMinZ);
	//// �E�B���h�E�T�C�Y(float)
	//POS3D_FLT sWndSize;
	//sWndSize.x = (obMax.x-obMin.x)	/ pixel_size + 100.0f;
	//sWndSize.y = (obMax.y-obMin.y)	/ pixel_size + 100.0f;
	//sWndSize.z = (fMaxZ-fMinZ)		/ pixel_size + 100.0f;
	// �E�B���h�E�T�C�Y(float)
	POS3D_FLT sWndSize;
	sWndSize.x = 256.0f;
	sWndSize.y = 256.0f;
	sWndSize.z = 256.0f;
	// �摜���S
	POS3D_FLT pxCenter;
	pxCenter.x = 0.5f*sWndSize.x;
	pxCenter.y = 0.5f*sWndSize.y;
	pxCenter.z = 0.5f*sWndSize.z;
	// �E�B���h�E�T�C�Y(int)
	int nWndSizeX = int(sWndSize.x);
	int nWndSizeY = int(sWndSize.y);
	int nWndSizeZ = int(sWndSize.z);

	// 3�����摜����
	//cv::Size img_sizeSD(nWndSizeX, nWndSizeZ);
	//cv::Size img_sizeFRNT(nWndSizeY, nWndSizeZ);
	//cv::Size img_sizeTP(nWndSizeX, nWndSizeY);
	cv::Size img_sizeSD(nWndSizeX, nWndSizeZ);
	cv::Size img_sizeFRNT(nWndSizeY, nWndSizeZ);
	cv::Size img_sizeTP(nWndSizeX/2, nWndSizeY/2);
	cv::Mat imgSD(img_sizeSD, CV_8UC3, cv::Scalar::all(255));
	cv::Mat imgFRNT(img_sizeFRNT, CV_8UC3, cv::Scalar::all(255));
	cv::Mat imgTP(img_sizeTP, CV_8UC3, cv::Scalar::all(255));

	POS3D_FLT delta;
	int nPixX, nPixY1, nPixY2, nPixZ;
	
	//float pixel_size = 0.005f;
	float pixel_size = max(max((obMax.x-obMin.x), (obMax.y-obMin.y)),(fMaxZ-fMinZ))/200.0f;

	for(int nIndex = 0; nIndex < pnt_size; ++nIndex){
		// ���S���W����̍�
		delta.x = (vecRotPntCld[nIndex].x-obCenter.x)	/ pixel_size;
		delta.y = (vecRotPntCld[nIndex].y-obCenter.y)	/ pixel_size;
		delta.z = (vecRotPntCld[nIndex].z-obCenter.z)	/ pixel_size;
		// �摜���W
		nPixX	= int(pxCenter.x + delta.x);
		nPixY1	= int(pxCenter.y + delta.y);
		nPixY2	= int(pxCenter.y - delta.y);
		nPixZ	= int(pxCenter.z - delta.z);
		//// �P�x�t�^
		//imgSD.at<Vec3b>(nPixZ, nPixX)		= (255, 0, 0);
		//imgFRNT.at<Vec3b>(nPixZ, nPixY1)	= (255, 0, 0);
		//imgTP.at<Vec3b>(nPixY2, nPixX)	= (255, 0, 0);
		// �P�x�t�^
		cv::circle(imgSD, cv::Point(nPixX, nPixZ), 3, color, -1, CV_AA);
		cv::circle(imgFRNT, cv::Point(nPixY1, nPixZ), 3, color, -1, CV_AA);
		cv::circle(imgTP, cv::Point(nPixX/2, nPixY2/2), 3, color, -1, CV_AA);
	}

	// �e�L�X�g��`��
	cv::Mat imgText(img_sizeTP, CV_8UC3, cv::Scalar::all(255));
	//int nPntX = int(pxCenter.x/4.0f);
	//int nPntY = int(pxCenter.y/4.0f);
	int nPntX = 17;
	int nPntY = 20;
	string strLine;
	while(getline(sstrText, strLine)){
		cv::putText(imgText, strLine, cv::Point(nPntX, nPntY), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1, CV_AA);
		nPntY += 15;
	}
	// �o��
//	cv::namedWindow("imageSD", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//	cv::namedWindow("imageFRNT", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//	cv::namedWindow("imageTP", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//	cv::namedWindow("imageText", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//	cv::imshow("imageSD", imgSD);
//	cv::imshow("imageFRNT", imgFRNT);
//	cv::imshow("imageTP", imgTP);
//	cv::imshow("imageText", imgText);

	cv::Mat combined_img(cv::Size(nWndSizeX+nWndSizeY+nWndSizeX/2, nWndSizeZ), CV_8UC3);
	cv::Rect roi_rect;

	roi_rect.width	= imgSD.cols;
	roi_rect.height = imgSD.rows;
	cv::Mat roi(combined_img, roi_rect);
	imgSD.copyTo(roi);
	roi_rect.x += imgSD.cols;

	roi_rect.width	= imgFRNT.cols;
	roi_rect.height = imgFRNT.rows;
	roi  = cv::Mat(combined_img, roi_rect);
	imgFRNT.copyTo(roi);
	roi_rect.x += imgFRNT.cols;

	roi_rect.width	= imgText.cols;
	roi_rect.height	= imgText.rows;
	roi  = cv::Mat(combined_img, roi_rect);
	imgText.copyTo(roi);
	roi_rect.y += imgText.rows;

	roi_rect.width	= imgTP.cols;
	roi_rect.height	= imgTP.rows;
	roi  = cv::Mat(combined_img, roi_rect);
	imgTP.copyTo(roi);

	cv::line(combined_img, cv::Point(nWndSizeX, 0), cv::Point(nWndSizeX, nWndSizeZ), cv::Scalar(0, 0, 0), 1, CV_AA);
	cv::line(combined_img, cv::Point(nWndSizeX+nWndSizeY, 0), cv::Point(nWndSizeX+nWndSizeY, nWndSizeZ), cv::Scalar(0, 0, 0), 1, CV_AA);
	cv::line(combined_img, cv::Point(nWndSizeX+nWndSizeY, nWndSizeZ/2), cv::Point(nWndSizeX+nWndSizeY+nWndSizeX/2, nWndSizeZ/2), cv::Scalar(0, 0, 0), 1, CV_AA);

//	cv::namedWindow("Combined Image", CV_WINDOW_AUTOSIZE);
//	cv::imshow("Combined Image", combined_img);
//	cv::waitKey(0);

	imwrite(strFileName, combined_img);
}

inline void CSegment::ImgCalar(int nLabel, cv::Scalar& color)
{
	switch(nLabel){
		case CAR:
			color = cv::Scalar(200, 40, 10);
			break;
		case PEDESTRIAN:
			color = cv::Scalar(30, 20, 180);
			break;
		case BICYCLIST:
			color = cv::Scalar(35, 115, 55);
			break;
		case BACKGROUND:
			color = cv::Scalar(150, 15, 75);
			break;
		default:
			color = cv::Scalar(200, 40, 10);
			break;
	}
}