#pragma once

#include	"RotatingCaliper.h"
#include	"feature.h"
#include	<windows.h>

using namespace	cv;

typedef	struct	OA_PARAM
{
	UINT	nMinPnts;							// �ŏ��_��
//	float	fContourEpsilon;					// Douglas-Peucker�ɂ������ߎ��������l
//	float	fMinBndBoxSize;						// ��]���Ȃ���`�̈�̍ŏ��T�C�Y
//	float	fMaxBndBoxSize;						// ��]���Ȃ���`�̈�̍ő�T�C�Y
}	FAR*	LPOA_PARAM;

class CObjAnalyze
{
protected:
	// --- �����o�ϐ�
	float			m_fMinZ, m_fMaxZ;			// �ő�ŏ�����
	vector<POS_FLT>	m_vecContour;				// �֊s(�ʕ���)
	RC_RECT			m_sMinAreaRect;				// �ʐύŏ��l�p�`
	vector<RC_RECT>	m_vecRotRect;
	vector<POS_FLT>	m_vecFront;					// �O�ʗ֊s

	// DEBUG
	vector<POS_FLT>	m_vecMerge;					// DEBUG

public:
	// --- �����o�֐�
	// �\�z�Ə���
	CObjAnalyze();
	~CObjAnalyze();

	// ���
	BOOL			Analyze(vector<POS3D_FLT>& vecPntCld);

	// �v�Z
	static CPosFlt	Conv(Point2f cSrc){	
		CPosFlt	cPos(cSrc.x,cSrc.y);	
		return	cPos;	
	};
	static Point2f	Conv(CPosFlt cSrc){	
		Point2f	cPos(cSrc.x,cSrc.y);	
		return	cPos;	
	};
	template<class POS_X>
		static void CreateCvPnts(vector<Point2f>* lpCvPnts, vector<POS_X>& vecPntCld);
	template<class POS_X>
		static void CreatePnts(vector<POS_X>* lpPntCld, vector<Point2f>& vecCvPnts);

	BOOL			CompRotateRectList(vector<RC_RECT>& vecRotRect, vector<POS_FLT>& vecPntCld/*, OA_PARAM sParam*/);
		
	BOOL			FindNrstAngleRect(LPRC_RECT lpRotRect, float fAngle);

	// ���擾
	float				GetMinZ(){			return	m_fMinZ;				};
	float				GetMaxZ(){			return	m_fMaxZ;				};
	RC_RECT				GetMinAreaRect(){	return	m_sMinAreaRect;			};
	vector<POS_FLT>&	GetContour(){		return	m_vecContour;			};
	vector<POS_FLT>&	GetFront(){			return	m_vecFront;				};
	vector<POS_FLT>&	GetMerge(){			return	m_vecMerge;				};
	vector<RC_RECT>&	GetRectCandidate(){	return	m_vecRotRect;			};		// Rotated Rect candidate

protected:
	// �����g�p�֐�
	template<class POS_X>
		BOOL	CompContour(vector<POS_FLT>* lpContour, vector<POS_X>& vecPntCld/*, float fEpsilon*/);
	BOOL	CompMinMaxZ(vector<POS3D_FLT>& vecPntCld);
	BOOL	CompFrontContour(vector<POS_FLT>& vecFront);
	BOOL	CompBackContour(vector<POS_FLT>& vecBack, vector<POS_FLT>& vecFront);
	BOOL	CompBndBox(/*OA_PARAM sParam*/);

};

// �C�����C���֐�
inline	CObjAnalyze::CObjAnalyze()
{
	m_fMinZ			= 0;
	m_fMaxZ			= 0;
	memset( &m_sMinAreaRect, 0x00, sizeof(m_sMinAreaRect) );
}

inline	CObjAnalyze::~CObjAnalyze()
{
}

inline	BOOL	CObjAnalyze::Analyze(vector<POS3D_FLT>& vecPntCld)
{
	// �L���ȃN���X�^���H
	if( vecPntCld.size()<2 )	
		return	FALSE;

	// �֊s�̌v�Z
	if( !CompContour(&m_vecContour, vecPntCld/*, sParam.fContourEpsilon*/) )
		return	FALSE;

	// ���֊s�̌v�Z
	if( !CompBndBox(/*sParam*/) )
		return	FALSE;

	// �ő�/�ŏ������̌v�Z
	if( !CompMinMaxZ(vecPntCld) )
		return	FALSE;

	return	TRUE;	
}

inline	BOOL	CObjAnalyze::CompBndBox(/*OA_PARAM sParam*/)
{
	// �O�ʗ֊s�̒��o
	vector<POS_FLT>	vecFront;
	CompFrontContour(vecFront);

	// ��ʗ֊s(�O�ʗ֊s��180�x��])�𒊏o
	vector<POS_FLT>	vecBack;
	CompBackContour(vecBack, vecFront);

	// �ʕ�̂ƌ�ʗ֊s�̍������쐬
	vector<POS_FLT>	vecMrgContour	= GetContour();									// �R�s�[�쐬
	vecMrgContour.insert( vecMrgContour.end(), vecBack.begin(), vecBack.end() );	// ��ʗ֊s�̍���

/*	// �����֊s���g����Bounding Box�̕����v�Z(L���`�󎞂̕������萸�x������)
	vector<Point2f>	vecCvPnts;
	CreateCvPnts(&vecCvPnts, vecMrgContour);
	RotatedRect	sRotRect	= minAreaRect(vecCvPnts);

	// �v�Z���ʂ̕ۑ�
	m_sMinAreaRect.sCenter	= CPoseFlt(sRotRect.center.x, sRotRect.center.y, Deg2Rad(sRotRect.angle));
	m_sMinAreaRect.sSize	= CPosFlt(sRotRect.size.width, sRotRect.size.height);	
	m_vecFront.swap(vecMrgContour);		// DEBUG
*/
	// �����֊s����l������Bounding Box�����v�Z
	vector<RC_RECT>	vecRotRect;
	if( !CompRotateRectList(vecRotRect, vecMrgContour/*, sParam*/) )
		return	FALSE;

	// �ʐύŏ���Bouding Box���擾
	float	fMinArea	= FLT_MAX;
	UINT	nMinAreaIndex;
	for(UINT nLoop=0; nLoop<vecRotRect.size(); nLoop++ )
	{
		float	fArea	= vecRotRect[nLoop].sSize.x * vecRotRect[nLoop].sSize.y;
		if( fArea<fMinArea ){
			fMinArea		= fArea;
			nMinAreaIndex	= nLoop;
		}
	}

	// �v�Z���ʂ̕ۑ�
	m_sMinAreaRect	= vecRotRect[nMinAreaIndex];
	m_vecRotRect.swap(vecRotRect);
//	m_vecFront.swap(vecMrgContour);		// DEBUG
	m_vecFront.swap(vecFront);			// DEBUG
	m_vecMerge.swap(vecMrgContour);		// DEBUG

	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompRotateRectList(vector<RC_RECT>& vecRotRect, vector<POS_FLT>& vecPntCld/*, OA_PARAM sParam*/)
{
	// �ʕ���(Convex Hull)�̌v�Z
	vector<POS_FLT>	vecContour;
	if( !CompContour(&vecContour, vecPntCld/*, -1.0f*/) )
		return	FALSE;

	CRotatingCaliper	cCaliper;
	if( !cCaliper.CompRectList(vecRotRect, vecContour/*, sParam.fMinBndBoxSize, sParam.fMaxBndBoxSize*/) )
//	if( !cCaliper.CompRectList(vecRotRect, vecContour) )
		return	FALSE;

	// DEBUG
/*	CString	strOutput;
	for( UINT nLoop=0; nLoop<vecRotRect.size(); nLoop++ )
	{
		float	fArea	= vecRotRect[nLoop].sSize.x * vecRotRect[nLoop].sSize.y;
		float	fAngle	= Rad2Deg(vecRotRect[nLoop].sCenter.psi);
		if( fAngle<0 )
			fAngle	+= 360;
		fAngle	= fmod( fAngle, 90.0f );
//		strOutput.Format("%f\t%f\n", Rad2Deg(vecRotRect[nLoop].sCenter.psi), fArea);
		strOutput.Format("%f\t%f\n", fAngle, 1/fArea);
		::OutputDebugStringA(strOutput);
	}
*/

	return	TRUE;
}

template<class POS_X>
inline	void CObjAnalyze::CreateCvPnts(vector<Point2f>* lpCvPnts, vector<POS_X>& vecPntCld)
{
	vector<Point2f>		vecCvPnts;
	vecCvPnts.reserve(vecPntCld.size());

	// OpenCV�p�Ɍ^�ϊ�
	vector<POS_X>::iterator	itrPntCld;
	for( itrPntCld=vecPntCld.begin(); itrPntCld!=vecPntCld.end(); itrPntCld++ )
		vecCvPnts.push_back( Conv(*itrPntCld) );

	// �ۑ�
	lpCvPnts->swap( vecCvPnts );
}

inline	BOOL	CObjAnalyze::FindNrstAngleRect(LPRC_RECT lpRotRect, float fAngle)
{
	if( m_vecRotRect.empty() )
		return	FALSE;

	// �p�x�����ł��������l�p�`��T��
	float	fMinDltAngle	= FLT_MAX;
	UINT	nMinDltIndex	= 0;
	float	fDltAngle;
	for( UINT nLoop=0; nLoop<m_vecRotRect.size(); nLoop++ )
	{
		// �p�x�΍����}90�x�ɐ��K�������Ƃ��̊p�x�����v�Z
		fDltAngle	= fmod( m_vecRotRect[nLoop].sCenter.psi-fAngle, 0.5f*(float)PAI );
		if( fDltAngle<-0.25f*PAI )		fDltAngle	+= 0.5*PAI;		// �p�x����-45�x��菬�������90�x�v���X
		else if( fDltAngle>+0.25f*PAI )	fDltAngle	-= 0.5*PAI;		// �p�x����+45�x���傫�����90�x�}�C�i�X

		// �ŏ��p�x���̃C���f�b�N�X��ۑ�
		fDltAngle	= fabs(fDltAngle);
		if( fDltAngle<fMinDltAngle ){
			fMinDltAngle	= fDltAngle;
			nMinDltIndex	= nLoop;
		}
	}

	// �ۑ�
	*lpRotRect	= m_vecRotRect[nMinDltIndex];

	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompFrontContour(vector<POS_FLT>& vecFront)
{
	vector<POS_FLT>&	vecContour	= GetContour();
	const UINT			nNrPnt		= vecContour.size();

	// �������𒊏o
	CPosFlt	cPos0, cPos1, cDlt;
	double	dSin, dCos, dLen;
	double	dRtvX, dRtvY;
	vector<BOOL>	vecIsFront(nNrPnt, FALSE);
	for(UINT nLoop=0; nLoop<nNrPnt; nLoop++)
	{
		// �����̒[�_���擾
		cPos0	= vecContour[nLoop];
		cPos1	= vecContour[(nLoop+1)%nNrPnt];

		// Sin, Cos
		cDlt	= cPos1 - cPos0;
		dLen	= hypot(cDlt.x, cDlt.y);
		dCos	= cDlt.x / dLen;
		dSin	= cDlt.y / dLen;

		// �ԗ����W������̍��[�_���猩�����W�ɕϊ�
		cDlt	= CPosFlt(0,0) - cPos0;
		dRtvX	=  cDlt.x*dCos + cDlt.y*dSin;
		dRtvY	= -cDlt.x*dSin + cDlt.y*dCos;
		if(dRtvY<0)
			vecIsFront[nLoop]	= TRUE;
	}

	// �����Ȃ��ӂ�T��
	UINT	nStrtIndex	= 0;
	while( nStrtIndex<vecIsFront.size() )
	{
		if( vecIsFront[nStrtIndex] )	
			nStrtIndex++;
		else	
			break;
	}

	// ������ӂ�T��
	while( nStrtIndex<vecIsFront.size() )
	{
		if( !vecIsFront[nStrtIndex] )
			nStrtIndex++;
		else
			break;
	}

	// ���_�Q�𒊏o
	vector<POS_FLT>	vecFrontContour;
	for(UINT nLoop=0; nLoop<nNrPnt; nLoop++)
	{
		int	nPos	= (nLoop+nStrtIndex)%vecIsFront.size();
		vecFrontContour.push_back( vecContour[nPos] );
		if( !vecIsFront[nPos] )
			break;
	}

	vecFront.swap(vecFrontContour);

	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompBackContour(vector<POS_FLT>& vecBack, vector<POS_FLT>& vecFront)
{
	if( vecFront.size()<3 )
		return	FALSE;

	// �s�����̒��_(��]���S�̌v�Z)
	CPosFlt	cCenter	= ( CPosFlt(vecFront.front()) + CPosFlt(vecFront.back()) )/2;

	// 180�x��]
	vector<POS_FLT>	vecBackContour;
	for( UINT nLoop=1; nLoop<vecFront.size()-1; nLoop++ )
	{
		CPosFlt	cNewPos	= cCenter*2 - vecFront[nLoop];
		vecBackContour.push_back(cNewPos);
	}

	// �ۑ�
	vecBack.swap(vecBackContour);
	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompMinMaxZ(vector<POS3D_FLT>& vecPntCld)
{
	float	fMinZ	= +FLT_MAX;
	float	fMaxZ	= -FLT_MAX;
	vector<POS3D_FLT>::iterator	itrPntCld;
	for( itrPntCld=vecPntCld.begin(); itrPntCld!=vecPntCld.end(); itrPntCld++ )
	{
		fMinZ	= min( fMinZ, itrPntCld->z );
		fMaxZ	= max( fMaxZ, itrPntCld->z );
	}

	// �ۑ�
	m_fMinZ	= fMinZ;
	m_fMaxZ	= fMaxZ;

	return	TRUE;
}

template<class POS_X>
inline	void CObjAnalyze::CreatePnts(vector<POS_X>* lpPntCld, vector<Point2f>& vecCvPnts)
{
	vector<POS_X>		vecPntCld;
	vecPntCld.reserve(vecCvPnts.size());

	// OpenCV�p�Ɍ^�ϊ�
	vector<Point2f>::iterator	itrCvPnt;
	for( itrCvPnt=vecCvPnts.begin(); itrCvPnt!=vecCvPnts.end(); itrCvPnt++ )
		vecPntCld.push_back( Conv(*itrCvPnt) );

	// �ۑ�
	lpPntCld->swap( vecPntCld );
}

template<class POS_X>
inline	BOOL	CObjAnalyze::CompContour(vector<POS_FLT>* lpContour, vector<POS_X>& vecPntCld/*, float fEpsilon*/)
{
	// OpenCV�p�Ɍ^�ϊ�
	vector<Point2f>	vecCvPnts;
	CreateCvPnts(&vecCvPnts, vecPntCld);
	
	// �ʕ���(Convex Hull)�̌v�Z
	vector<Point2f>		vecCvContour;
	convexHull(vecCvPnts, vecCvContour);	

	// Douglas-Peucker�ɂ������ߎ�(�����̍������̂��߁C�������_�����팸)	�� ���ۂɂ͍������ɂ��܂��^���Ȃ�����
/*	if(fEpsilon>0)
	{
		vector<Point2f>		vecApprox;
		approxPolyDP(vecCvContour, vecApprox, fEpsilon, TRUE);

		vecCvContour.swap(vecApprox);
	}
*/	
	// �`���͎��s
	const UINT	nNrPoint	= vecCvContour.size();
	if( nNrPoint<2 )
		return	FALSE;

	// �^�ϊ�
	vector<POS_FLT>	vecContour;
	CreatePnts(&vecContour, vecCvContour);

	// �ۑ�
	lpContour->swap(vecContour);
	return	TRUE;
}