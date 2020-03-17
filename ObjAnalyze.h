#pragma once

#include	"RotatingCaliper.h"
#include	"feature.h"
#include	<windows.h>

using namespace	cv;

typedef	struct	OA_PARAM
{
	UINT	nMinPnts;							// 最小点数
//	float	fContourEpsilon;					// Douglas-Peuckerによる線分近似しきい値
//	float	fMinBndBoxSize;						// 回転しない矩形領域の最小サイズ
//	float	fMaxBndBoxSize;						// 回転しない矩形領域の最大サイズ
}	FAR*	LPOA_PARAM;

class CObjAnalyze
{
protected:
	// --- メンバ変数
	float			m_fMinZ, m_fMaxZ;			// 最大最少高さ
	vector<POS_FLT>	m_vecContour;				// 輪郭(凸方体)
	RC_RECT			m_sMinAreaRect;				// 面積最小四角形
	vector<RC_RECT>	m_vecRotRect;
	vector<POS_FLT>	m_vecFront;					// 前面輪郭

	// DEBUG
	vector<POS_FLT>	m_vecMerge;					// DEBUG

public:
	// --- メンバ関数
	// 構築と消滅
	CObjAnalyze();
	~CObjAnalyze();

	// 解析
	BOOL			Analyze(vector<POS3D_FLT>& vecPntCld);

	// 計算
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

	// 情報取得
	float				GetMinZ(){			return	m_fMinZ;				};
	float				GetMaxZ(){			return	m_fMaxZ;				};
	RC_RECT				GetMinAreaRect(){	return	m_sMinAreaRect;			};
	vector<POS_FLT>&	GetContour(){		return	m_vecContour;			};
	vector<POS_FLT>&	GetFront(){			return	m_vecFront;				};
	vector<POS_FLT>&	GetMerge(){			return	m_vecMerge;				};
	vector<RC_RECT>&	GetRectCandidate(){	return	m_vecRotRect;			};		// Rotated Rect candidate

protected:
	// 内部使用関数
	template<class POS_X>
		BOOL	CompContour(vector<POS_FLT>* lpContour, vector<POS_X>& vecPntCld/*, float fEpsilon*/);
	BOOL	CompMinMaxZ(vector<POS3D_FLT>& vecPntCld);
	BOOL	CompFrontContour(vector<POS_FLT>& vecFront);
	BOOL	CompBackContour(vector<POS_FLT>& vecBack, vector<POS_FLT>& vecFront);
	BOOL	CompBndBox(/*OA_PARAM sParam*/);

};

// インライン関数
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
	// 有効なクラスタか？
	if( vecPntCld.size()<2 )	
		return	FALSE;

	// 輪郭の計算
	if( !CompContour(&m_vecContour, vecPntCld/*, sParam.fContourEpsilon*/) )
		return	FALSE;

	// 可視輪郭の計算
	if( !CompBndBox(/*sParam*/) )
		return	FALSE;

	// 最大/最少高さの計算
	if( !CompMinMaxZ(vecPntCld) )
		return	FALSE;

	return	TRUE;	
}

inline	BOOL	CObjAnalyze::CompBndBox(/*OA_PARAM sParam*/)
{
	// 前面輪郭の抽出
	vector<POS_FLT>	vecFront;
	CompFrontContour(vecFront);

	// 後面輪郭(前面輪郭を180度回転)を抽出
	vector<POS_FLT>	vecBack;
	CompBackContour(vecBack, vecFront);

	// 凸包体と後面輪郭の合成を作成
	vector<POS_FLT>	vecMrgContour	= GetContour();									// コピー作成
	vecMrgContour.insert( vecMrgContour.end(), vecBack.begin(), vecBack.end() );	// 後面輪郭の合成

/*	// 合成輪郭を使ってBounding Boxの方向計算(L字形状時の方向推定精度を向上)
	vector<Point2f>	vecCvPnts;
	CreateCvPnts(&vecCvPnts, vecMrgContour);
	RotatedRect	sRotRect	= minAreaRect(vecCvPnts);

	// 計算結果の保存
	m_sMinAreaRect.sCenter	= CPoseFlt(sRotRect.center.x, sRotRect.center.y, Deg2Rad(sRotRect.angle));
	m_sMinAreaRect.sSize	= CPosFlt(sRotRect.size.width, sRotRect.size.height);	
	m_vecFront.swap(vecMrgContour);		// DEBUG
*/
	// 合成輪郭から考えられるBounding Box候補を計算
	vector<RC_RECT>	vecRotRect;
	if( !CompRotateRectList(vecRotRect, vecMrgContour/*, sParam*/) )
		return	FALSE;

	// 面積最小のBouding Boxを取得
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

	// 計算結果の保存
	m_sMinAreaRect	= vecRotRect[nMinAreaIndex];
	m_vecRotRect.swap(vecRotRect);
//	m_vecFront.swap(vecMrgContour);		// DEBUG
	m_vecFront.swap(vecFront);			// DEBUG
	m_vecMerge.swap(vecMrgContour);		// DEBUG

	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompRotateRectList(vector<RC_RECT>& vecRotRect, vector<POS_FLT>& vecPntCld/*, OA_PARAM sParam*/)
{
	// 凸方体(Convex Hull)の計算
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

	// OpenCV用に型変換
	vector<POS_X>::iterator	itrPntCld;
	for( itrPntCld=vecPntCld.begin(); itrPntCld!=vecPntCld.end(); itrPntCld++ )
		vecCvPnts.push_back( Conv(*itrPntCld) );

	// 保存
	lpCvPnts->swap( vecCvPnts );
}

inline	BOOL	CObjAnalyze::FindNrstAngleRect(LPRC_RECT lpRotRect, float fAngle)
{
	if( m_vecRotRect.empty() )
		return	FALSE;

	// 角度差が最も小さい四角形を探索
	float	fMinDltAngle	= FLT_MAX;
	UINT	nMinDltIndex	= 0;
	float	fDltAngle;
	for( UINT nLoop=0; nLoop<m_vecRotRect.size(); nLoop++ )
	{
		// 角度偏差を±90度に正規化したときの角度差を計算
		fDltAngle	= fmod( m_vecRotRect[nLoop].sCenter.psi-fAngle, 0.5f*(float)PAI );
		if( fDltAngle<-0.25f*PAI )		fDltAngle	+= 0.5*PAI;		// 角度差が-45度より小さければ90度プラス
		else if( fDltAngle>+0.25f*PAI )	fDltAngle	-= 0.5*PAI;		// 角度差が+45度より大きければ90度マイナス

		// 最小角度差のインデックスを保存
		fDltAngle	= fabs(fDltAngle);
		if( fDltAngle<fMinDltAngle ){
			fMinDltAngle	= fDltAngle;
			nMinDltIndex	= nLoop;
		}
	}

	// 保存
	*lpRotRect	= m_vecRotRect[nMinDltIndex];

	return	TRUE;
}

inline	BOOL	CObjAnalyze::CompFrontContour(vector<POS_FLT>& vecFront)
{
	vector<POS_FLT>&	vecContour	= GetContour();
	const UINT			nNrPnt		= vecContour.size();

	// 可視線分を抽出
	CPosFlt	cPos0, cPos1, cDlt;
	double	dSin, dCos, dLen;
	double	dRtvX, dRtvY;
	vector<BOOL>	vecIsFront(nNrPnt, FALSE);
	for(UINT nLoop=0; nLoop<nNrPnt; nLoop++)
	{
		// 線分の端点を取得
		cPos0	= vecContour[nLoop];
		cPos1	= vecContour[(nLoop+1)%nNrPnt];

		// Sin, Cos
		cDlt	= cPos1 - cPos0;
		dLen	= hypot(cDlt.x, cDlt.y);
		dCos	= cDlt.x / dLen;
		dSin	= cDlt.y / dLen;

		// 車両座標を線分の左端点から見た座標に変換
		cDlt	= CPosFlt(0,0) - cPos0;
		dRtvX	=  cDlt.x*dCos + cDlt.y*dSin;
		dRtvY	= -cDlt.x*dSin + cDlt.y*dCos;
		if(dRtvY<0)
			vecIsFront[nLoop]	= TRUE;
	}

	// 見えない辺を探索
	UINT	nStrtIndex	= 0;
	while( nStrtIndex<vecIsFront.size() )
	{
		if( vecIsFront[nStrtIndex] )	
			nStrtIndex++;
		else	
			break;
	}

	// 見える辺を探索
	while( nStrtIndex<vecIsFront.size() )
	{
		if( !vecIsFront[nStrtIndex] )
			nStrtIndex++;
		else
			break;
	}

	// 可視点群を抽出
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

	// 不可視部の中点(回転中心の計算)
	CPosFlt	cCenter	= ( CPosFlt(vecFront.front()) + CPosFlt(vecFront.back()) )/2;

	// 180度回転
	vector<POS_FLT>	vecBackContour;
	for( UINT nLoop=1; nLoop<vecFront.size()-1; nLoop++ )
	{
		CPosFlt	cNewPos	= cCenter*2 - vecFront[nLoop];
		vecBackContour.push_back(cNewPos);
	}

	// 保存
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

	// 保存
	m_fMinZ	= fMinZ;
	m_fMaxZ	= fMaxZ;

	return	TRUE;
}

template<class POS_X>
inline	void CObjAnalyze::CreatePnts(vector<POS_X>* lpPntCld, vector<Point2f>& vecCvPnts)
{
	vector<POS_X>		vecPntCld;
	vecPntCld.reserve(vecCvPnts.size());

	// OpenCV用に型変換
	vector<Point2f>::iterator	itrCvPnt;
	for( itrCvPnt=vecCvPnts.begin(); itrCvPnt!=vecCvPnts.end(); itrCvPnt++ )
		vecPntCld.push_back( Conv(*itrCvPnt) );

	// 保存
	lpPntCld->swap( vecPntCld );
}

template<class POS_X>
inline	BOOL	CObjAnalyze::CompContour(vector<POS_FLT>* lpContour, vector<POS_X>& vecPntCld/*, float fEpsilon*/)
{
	// OpenCV用に型変換
	vector<Point2f>	vecCvPnts;
	CreateCvPnts(&vecCvPnts, vecPntCld);
	
	// 凸方体(Convex Hull)の計算
	vector<Point2f>		vecCvContour;
	convexHull(vecCvPnts, vecCvContour);	

	// Douglas-Peuckerによる線分近似(処理の高速化のため，線分頂点数を削減)	⇒ 実際には高速化にあまり寄与しなかった
/*	if(fEpsilon>0)
	{
		vector<Point2f>		vecApprox;
		approxPolyDP(vecCvContour, vecApprox, fEpsilon, TRUE);

		vecCvContour.swap(vecApprox);
	}
*/	
	// 形状解析失敗
	const UINT	nNrPoint	= vecCvContour.size();
	if( nNrPoint<2 )
		return	FALSE;

	// 型変換
	vector<POS_FLT>	vecContour;
	CreatePnts(&vecContour, vecCvContour);

	// 保存
	lpContour->swap(vecContour);
	return	TRUE;
}