//////////////////////////////////////////////////////////////////////////////////////////////////
//		ObjClassify.hへ移行
//////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include	<fstream>
#include	"ObjTrck.h"

#define CLASS_NUM		4		//クラス数
#define LEARN_NUM_TRCK	73		//学習回数 - トラック識別	// ★	JARI用の設定58

#define PROB_DIM		2		//確率密度分布の次元数

const int BIN[PROB_DIM]	= {64, 20};							// ★
const int PREDICT_NUM	= BIN[0]*BIN[1];

#define PIXEL_SIZE		0.06f

#define COVMTRX_ELNUM	6		//3次元共分散行列要素数
#define INRMTRX_ELNUM	6		//3次元慣性モーメント行列要素数

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

// 特徴量リスト
enum Feature
{
	//大きさ
	LONGSIDE,
	NARROWSIDE,
	HEIGHT,
	//ポイント数
	POINTNUM,
	//反射強度平均・分散値
	INTENSITY,
	VARINTENSITY,
	//凸包体（側面）
	SIDEAREA,
	SIDEPER,
	//凸包体（正面）
	FRONTAREA,
	FRONTPER,
	//凸包体（上面）
	TOPAREA,
	TOPPER,
	//2値画像面積
	SD_NONZERO,
	FRNT_NONZERO,
	TP_NONZERO,
	//掛け合わせ
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
	//3次元共分散行列
	COVMTRXSTRT,
	COVMTRXEND = COVMTRXSTRT + COVMTRX_ELNUM - 1,
	//3次元慣性モーメント行列
	INRMTRXSTRT,
	INRMTRXEND = INRMTRXSTRT + INRMTRX_ELNUM - 1,
	//ポイントクラウド分布
	PNTHSTLNGSTRT,
	PNTHSTLNGEND = PNTHSTLNGSTRT + PNTHST_LNGDIM - 1,
	PNTHSTNRRWSTRT,
	PNTHSTNRRWEND = PNTHSTNRRWSTRT + PNTHST_NRRWDIM - 1,
	//反射強度分布
	INTHSTSTRT,
	INTHSTEND = INTHSTSTRT + INTHST_DIM - 1,
	//最大・最小反射強度
	INTMAX,
	INTMIN,
	//スライス特徴量
	SLICE_LWSTRT,
	SLICE_LWEND = SLICE_LWSTRT + LAYERNUM - 1,
	SLICE_NWSTRT,
	SLICE_NWEND = SLICE_NWSTRT + LAYERNUM - 1,
	SLICE_AISTRT,
	SLICE_AIEND = SLICE_AISTRT + LAYERNUM - 1,
	SLICE_PAI,
	SLICE_PVI,
	//速度
	VELOCITY,
	//速度の分散
	VELVAR,

	// 面積最小矩形
	BOXAREA_TP,							// ★
	// Box体積 
	MINBOXVOL,							// ★

	DIM_NUM,
	
	//地面からの最大・最小高さ
	MAXHEIGHT,
	MINHEIGHT,
	//自車からの距離
	DISTANCE,

	ALLFTR_NUM
};


//識別器構造体
struct Classifier
{
	vector<int> HistDim;						//特徴量の次元
	vector<double> Lut;							//確率密度分布
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
// 識別クラス
class CObjClassify
{
private:
	// --- メンバ変数
	vector<float>			m_vecFeature;
	vector<double>			m_vecClScore;
	vector<double>			m_vecClProb;

	vector<int>				m_vecFtrList;
	vector<Classifier>		m_vecWeakCl;
	vector<vector<float>>	m_vecSplitPoint;
	vector<vector<double>>	m_vecSgmdParam;

public:
	// --- メンバ関数
	// 構築と消滅
	CObjClassify();
	~CObjClassify();

	// 初期化
	void			Init(string& strClssfrFlNm, string& strBinFlNm, string& strSgmdPrmFlNm);

	BOOL			AnalyzeFeature(vector<POSINT_FLT>& vecPntCld, POSE_DBL sPose, CObjAnalyze& cObj, CObjTrck& cTrck);
	void			ClassifyRAB(vector<int>& vecFtrList, vector<Classifier>& vecWeakCl, vector<vector<float>>& vecSplitPoint, vector<vector<double>>& vecSgmdParam);
//	static int		UpdateClass(vector<double> vecClProb);
	static BOOL		FuncClassiy(vector<double>& vecProb, POSE_DBL sPose, CObjTrck& cTrck, CObjAnalyze& cObj, vector<POSINT_FLT>& vecPntCld, LPVOID lpData){		
			return	((CObjClassify*)lpData)->FuncClassiy(vecProb, sPose, cTrck, cObj, vecPntCld);		
	};

	// 計算
	UINT			CompLernNum(string& strClssfrFlNm);


	// 情報取得
	vector<double>&			GetClScore(){		return	m_vecClScore;		};
	vector<double>&			GetClProb(){		return	m_vecClProb;		};
	vector<float>&			GetFeature(){		return	m_vecFeature;		};
	vector<int>&			GetFtrList(){		return	m_vecFtrList;		};
	vector<Classifier>&		GetWeakCl(){		return	m_vecWeakCl;		};
	vector<vector<float>>&	GetSplitPoint(){	return	m_vecSplitPoint;	};
	vector<vector<double>>&	GetSigmoidParam(){	return	m_vecSgmdParam;		};

private:
	// 内部使用関数
	void	CreateFtrStrngth(vector<float>& vecClFeature, vector<vector<UINT>>& vecFtrStrngth,  vector<vector<float>>& vecSplitPoint);
	void	Prediction(vector<vector<UINT>>& vecFtrStrngth, vector<Classifier>& vecWeakCl);
	void	Score2Prob(vector<double>& vecClScore, vector<vector<double>>& vecSgmdParam);

	// 特徴量計算
	void	CompObjInf(LPOC_OBJ_INF lpObjInf, vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj);		// 旧SetParam関数
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

	// 識別実行
	BOOL	FuncClassiy(vector<double>& vecProb, POSE_DBL sPose, CObjTrck& cTrck, CObjAnalyze& cObj, vector<POSINT_FLT>& vecPntCld);

	// ファイル読込
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

	///////////////////////////////////////Track用識別器///////////////////////////////////////
	// 特徴量リスト
	vecFtrList.resize(DIM_NUM);
	for(int nDim = 0; nDim < DIM_NUM; ++nDim){
		vecFtrList[nDim] = nDim;
	}
	nDimNum			= vecFtrList.size();
//	nLearnNum		= LEARN_NUM_TRCK;
	
	//メモリ確保
//	m_vecWeakCl		= vector<Classifier>(nLearnNum * CLASS_NUM);
//	m_vecSplitPoint	= vector<vector<float>>(nDimNum, vector<float>(BIN[0]+BIN[1]));
//	m_vecSgmdParam	= vector<vector<double>>(CLASS_NUM, vector<double>(2));

	//読み込み
	ReadCl_RAB(m_vecWeakCl, m_vecSplitPoint, m_vecSgmdParam, strClssfrFlNm, strBinFlNm, strSgmdPrmFlNm);
	
	//保存
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
	// 学習回数をファイルサイズから計算
	UINT	nLearnNum	= CompLernNum(strClssfrFlNm);

	//メモリ確保
	vecWeakCl		= vector<Classifier>(nLearnNum * CLASS_NUM);
	vecSplitPoint	= vector<vector<float>>(DIM_NUM, vector<float>(BIN[0]+BIN[1]));
	vecSgmdParam	= vector<vector<double>>(CLASS_NUM, vector<double>(2));

	// 弱識別器
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

	// BINの幅
	fstream ReadBin;
	ReadBin.open(strBinFlNm, ios::in | ios::binary);
	if (ReadBin.is_open()){
		//特徴量次元数
		UINT nDimNum = vecSplitPoint.size();
		for(UINT nDim = 0; nDim < nDimNum; ++nDim){
			ReadBin.read((char*)&vecSplitPoint[nDim][0], sizeof(float)*(BIN[0]+BIN[1]));
		}
		ReadBin.close();
	}

	// シグモイド関数のパラメータ
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
	// メモリ確保
	m_vecFeature = vector<float>(ALLFTR_NUM);

	// 追跡物体情報の取得
	float	fVelocity	= cTrck.CompVelocity();		// 物体速度
	float	fVelVar		= cTrck.CompVelVar();		// 速度分散
	float	fPsi		= cTrck.GetTrckAngle();		// 物体方位

	//ポイント数
	UINT	pnt_size		= vecPntCld.size();
	m_vecFeature[POINTNUM]	= (float)pnt_size;

	//反射強度を0-1から0-255に変換
	vector<float> vecIntValue(pnt_size);
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		vecIntValue[nIndex] = 255.0f * vecPntCld[nIndex].i;
	}
	
	//Trackの中心姿勢
	CPoseDbl	cTrckPose(cTrck.GetTrckPose());
	cTrckPose.Abs2Rtv(sPose);
	POSE_FLT	sTrckPose = ConvPose(cTrckPose);

	//ポイントクラウドの座標回転
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

	// 物体情報の予備計算
	OC_OBJ_INF	sObjInf;
	CompObjInf(&sObjInf, vecLngPnt, vecNrrwPnt, vecHghtPnt, cObj);

	//大きさに関する特徴
	CompSizeFtr(sObjInf);
	//反射強度の平均，分散
	AnalyzeIntensity(vecIntValue);
	//形状に関する特徴量（凸包体3面）
	AnalyzeConvexHull(vecLngPnt, vecNrrwPnt, vecHghtPnt, cObj);
	//Box情報から上面凸包体のみ利用
//	AnalyzeConvexHull(cObj);
	//2値画像における面積
	AnalyzeBinImg(vecLngPnt, vecNrrwPnt, vecHghtPnt, sObjInf);
	//3次元共分散行列，3次元慣性モーメント行列
	CompCovMtrx(vecLngPnt, vecNrrwPnt, vecHghtPnt, sObjInf);
	//ポイントクラウド分布
	AnalyzePntHist(vecLngPnt, vecNrrwPnt, sObjInf);
	//反射強度分布
	AnalyzeIntHist(vecIntValue);
	//スライス特徴量
	AnalyzeSlice(vecLngPnt, vecNrrwPnt, vecHghtPnt, vecIntValue);
	//掛け合わせ
	CompMultiFtr();

	// 車速
	if( cTrck.GetTrckState()!=CObjTrck::stateConfirmed)
		m_vecFeature[VELOCITY]	= 0;
	else
		m_vecFeature[VELOCITY]	= fVelocity;

	//m_vecFeature[VELVAR]	= fVelVar;
	m_vecFeature[VELVAR]	= POW_SQ(fVelocity)/fVelVar;		// ★

	// 追加	// ★
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

	// X,Yの最大・最小値
	POS_FLT	sMax = CPosFlt(-FLT_MAX, -FLT_MAX);
	POS_FLT sMin = CPosFlt(FLT_MAX, FLT_MAX);
	for( UINT nIndex=0; nIndex<pnt_size; nIndex++ )
	{
		sMax.x	= max( sMax.x, vecLngPnt[nIndex]	);
		sMax.y	= max( sMax.y, vecNrrwPnt[nIndex]	);
		sMin.x	= min( sMin.x, vecLngPnt[nIndex]	);
		sMin.y	= min( sMin.y, vecNrrwPnt[nIndex]	);
	}

	// 点群の重心
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

	//保存
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
	// 幅・高さ
	m_vecFeature[LONGSIDE]		= sObjInf.sMax.x - sObjInf.sMin.x;
	m_vecFeature[NARROWSIDE]	= sObjInf.sMax.y - sObjInf.sMin.y;
	m_vecFeature[HEIGHT]		= sObjInf.sMax.z - sObjInf.sMin.z;

	// 最大・最小高さ
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
		//反射強度が有効なものだけを使用
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
		//平均値
		fAvgInt = fAvgInt/(float)count;
		//分散値
		fVarInt = fSqInt/(float)count - fAvgInt*fAvgInt;
	}

	m_vecFeature[INTENSITY]		= fAvgInt;
	m_vecFeature[VARINTENSITY]	= fVarInt;

	//最大・最小値
	m_vecFeature[INTMAX]		= fMaxInt;
	m_vecFeature[INTMIN]		= fMinInt;
}

inline	void	CObjClassify::AnalyzeConvexHull(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, CObjAnalyze& cObj)
{
	UINT	pnt_size = vecLngPnt.size();

	//OpennCV用に型変換
	vector<Point2f>	vecSdCvPnts(pnt_size);
	vector<Point2f>	vecFrntCvPnts(pnt_size);
	for (UINT nLoop = 0; nLoop < pnt_size; nLoop++)
	{
		vecSdCvPnts[nLoop]		= Point2f(vecLngPnt[nLoop], vecHghtPnt[nLoop]);
		vecFrntCvPnts[nLoop]	= Point2f(vecNrrwPnt[nLoop], vecHghtPnt[nLoop]);
	}
	//正面・側面
	vector<Point2f>		vecSdConvexHull;
	vector<Point2f>		vecFrntConvexHull;
	convexHull(vecSdCvPnts, vecSdConvexHull);
	convexHull(vecFrntCvPnts, vecFrntConvexHull);

	//上面はBox情報を利用
	vector<Point2f>		vecTpConvexHull;
	vector<POS_FLT>&	vecContour = cObj.GetContour();
	vecTpConvexHull.reserve(vecContour.size());
	vector<POS_FLT>::iterator	itrContour;
	for( itrContour=vecContour.begin(); itrContour!=vecContour.end(); itrContour++ )
		vecTpConvexHull.push_back( cObj.Conv(*itrContour) );

	//面積
	double SdArea		= contourArea(vecSdConvexHull);
	double FrntArea		= contourArea(vecFrntConvexHull);
	double TpArea		= contourArea(vecTpConvexHull);
	//周囲長さ
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
	//上面はBox情報を利用
	vector<Point2f>		vecTpConvexHull;
	vector<POS_FLT>&	vecContour = cObj.GetContour();
	vecTpConvexHull.reserve(vecContour.size());
	vector<POS_FLT>::iterator	itrContour;
	for( itrContour=vecContour.begin(); itrContour!=vecContour.end(); itrContour++ )
		vecTpConvexHull.push_back( cObj.Conv(*itrContour) );

	//面積
	double TpArea		= contourArea(vecTpConvexHull);
	//周囲長さ
	double TpLength		= arcLength(vecTpConvexHull,true);

	m_vecFeature[TOPAREA]		= (float)TpArea;
	m_vecFeature[TOPPER]		= (float)TpLength;
}

inline	void	CObjClassify::AnalyzeBinImg(vector<float>& vecLngPnt, vector<float>& vecNrrwPnt, vector<float>& vecHghtPnt, OC_OBJ_INF sObjInf)
{
	UINT	pnt_size	= vecLngPnt.size();

	//X,Y,Zの中心座標
	POS3D_FLT obCenter;
	obCenter.x = 0.5f*(sObjInf.sMax.x	+ sObjInf.sMin.x);
	obCenter.y = 0.5f*(sObjInf.sMax.y	+ sObjInf.sMin.y);
	obCenter.z = 0.5f*(sObjInf.sMax.z	+ sObjInf.sMin.z);
	//ウィンドウサイズ(float)
	POS3D_FLT fWindSize;
	fWindSize.x = (sObjInf.sMax.x	- sObjInf.sMin.x)	/ PIXEL_SIZE + 10.0f;
	fWindSize.y = (sObjInf.sMax.y	- sObjInf.sMin.y)	/ PIXEL_SIZE + 10.0f;
	fWindSize.z = (sObjInf.sMax.z	- sObjInf.sMin.z)	/ PIXEL_SIZE + 10.0f;
	//ウィンドウサイズ(int)
	int iWindSizeX = int(0.5f+fWindSize.x);
	int iWindSizeY = int(0.5f+fWindSize.y);
	int iWindSizeZ = int(0.5f+fWindSize.z);
	//正面・側面・上面画像生成
	cv::Mat Side_img	= cv::Mat::zeros(cv::Size(iWindSizeX, iWindSizeZ), CV_8UC1);
	cv::Mat Front_img	= cv::Mat::zeros(cv::Size(iWindSizeY, iWindSizeZ), CV_8UC1);
	cv::Mat Top_img		= cv::Mat::zeros(cv::Size(iWindSizeX, iWindSizeY), CV_8UC1); 
	//画像中心
	POS3D_FLT pxCenter;
	pxCenter.x = 0.5f * fWindSize.x;
	pxCenter.y = 0.5f * fWindSize.y;
	pxCenter.z = 0.5f * fWindSize.z;

	POS3D_FLT delta;
	int iPixX, iPixY1, iPixY2, iPixZ;
	for (UINT i = 0; i < pnt_size; ++i){
		//中心座標からの差
		delta.x = (vecLngPnt[i]		- obCenter.x) / PIXEL_SIZE;
		delta.y = (vecNrrwPnt[i]	- obCenter.y) / PIXEL_SIZE;
		delta.z = (vecHghtPnt[i]	- obCenter.z) / PIXEL_SIZE;
		//画像上の座標
		iPixX	= int(pxCenter.x + delta.x);
		iPixY1	= int(pxCenter.y + delta.y);
		iPixY2	= int(pxCenter.y - delta.y);
		iPixZ	= int(pxCenter.z - delta.z);
		//投影
		Side_img.at<uchar>(iPixZ, iPixX)	= 255;
		Front_img.at<uchar>(iPixZ, iPixY1)	= 255;
		Top_img.at<uchar>(iPixY2, iPixX)	= 255;
	}
	//投影された領域を計算（面積）
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
	vector<float> vecCovElmnt(COVMTRX_ELNUM, 0.0f);		//3次元共分散行列
	vector<float> vecInrElmnt(INRMTRX_ELNUM, 0.0f);		//3次元慣性モーメント行列
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		//重心を中心とした座標
		sPosDlt.x = vecLngPnt[nIndex]	- sGrav.x;
		sPosDlt.y = vecNrrwPnt[nIndex]	- sGrav.y;
		sPosDlt.z = vecHghtPnt[nIndex]	- sGrav.z;
		//3次元共分散行列
		vecCovElmnt[0] += sPosDlt.x * sPosDlt.x;
		vecCovElmnt[1] += sPosDlt.x * sPosDlt.y;
		vecCovElmnt[2] += sPosDlt.x * sPosDlt.z;
		vecCovElmnt[3] += sPosDlt.y * sPosDlt.y;
		vecCovElmnt[4] += sPosDlt.y * sPosDlt.z;
		vecCovElmnt[5] += sPosDlt.z * sPosDlt.z;
		//3次元慣性モーメント行列
		vecInrElmnt[0] += sPosDlt.y * sPosDlt.y + sPosDlt.z * sPosDlt.z;
		vecInrElmnt[1] -= sPosDlt.x * sPosDlt.y;
		vecInrElmnt[2] -= sPosDlt.x * sPosDlt.z;
		vecInrElmnt[3] += sPosDlt.x * sPosDlt.x + sPosDlt.z * sPosDlt.z;
		vecInrElmnt[4] -= sPosDlt.y * sPosDlt.z;
		vecInrElmnt[5] += sPosDlt.x * sPosDlt.x + sPosDlt.y * sPosDlt.y;
	}
	//3次元共分散行列のみ正規化
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
	//側面ポイントクラウド分布(ヒストグラム)
	CompPntHist(vecLngPntHist, vecLngPnt, fLngGrav, fBlckSize, fWdthRng);

	fBlckSize	= PNTHST_PTCH;
	fWdthRng	= PNTHST_NRRWRNG;
	vector<int> vecNrrwPntHist;
	//正面ポイントクラウド分布(ヒストグラム)
	CompPntHist(vecNrrwPntHist, vecNrrwPnt, fNrrwGrav, fBlckSize, fWdthRng);

	//正規化して保存
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

	int			nLoopNum	= int(0.5 + fWdthRng/fBlckSize);	//水平方向分割数(四捨五入)
	int			tmp			= int(0.5 * nLoopNum);
	//分割点最小値
	float		fMinHrz		= fGrav - 0.5f*fBlckSize - fBlckSize*(float)tmp;		

	int			num;
	float		fDlt, fRate;
	vector<int>	vecHist(nLoopNum, 0);
	for (UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		//最小値との差
		fDlt	= vecPnt[nIndex] - fMinHrz;
		//全幅に対する割合0-1
		fRate	= fDlt	/ fWdthRng;
		//BINの番号
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
	//BIN数(四捨五入)
	int				nLoopNum	= INTHST_DIM;

	int				num;
	UINT			nCnt = 0;
	float			fValue;
	float			fRate;
	vector<UINT>	vecIntHist(nLoopNum, 0);
	for (UINT nIndex = 0; nIndex < pnt_size-1; ++nIndex){
		fValue = vecIntValue[nIndex];
		//反射強度が有効なものだけを使用
		if(fValue > 0.0f){
			//全幅に対する割合0-1
			fRate	= fValue / fMaxVal;
			//BINの番号
			num		= int(fRate * nLoopNum);
			if(num > nLoopNum-1)
				num = nLoopNum -1;

			++vecIntHist[num];
			++nCnt;
		}
	}

	if(nCnt > 0){
		//正規化
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
	//ポイントクラウドを分割
	CompLayer(vecHghtPnt, vecPartNum);
	//パーツ内の幅(長辺)
	vector<float>	vecLngWdth(nLoopNum);
	CompPartWdth(vecLngPnt, vecLngWdth, vecPartNum);
	//パーツ内の幅(短辺)
	vector<float>	vecNrrwWdth(nLoopNum);
	CompPartWdth(vecNrrwPnt, vecNrrwWdth, vecPartNum);
	//パーツ内の平均反射強度
	vector<float>	vecAvrgInt(nLoopNum, 0.0f);
	AnalyzePartInt(vecIntValue, vecAvrgInt, vecPartNum);

	float	fPartAvrgInt	= 0.0f;	//全パーツにおける平均反射強度
	float	fPartSqInt		= 0.0f;	
	float	fPartVarInt		= 0.0f;	//全パーツにおける反射強度分散
	float	fValue;
	UINT	nCnt			= 0;
	for(UINT nLoop = 0; nLoop < nLoopNum; ++nLoop){
		fValue = vecAvrgInt[nLoop];
		//反射強度が有効なものだけを使用
		if(fValue > 0.0f){
			fPartAvrgInt	+= fValue;
			fPartSqInt		+= fValue * fValue;
			++nCnt;
		}
	}
	if(nCnt > 0){
		//平均値
		fPartAvrgInt /= (float)nCnt;
		//分散値
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

	UINT	nLoopNum	= LAYERNUM;						//レイヤーの数
	float	fWdith		= SLICE_MAXHGHT-SLICE_MINHGHT;	//全幅
	float	fMinHght	= SLICE_MINHGHT;				//高さの最小値
	
	int		num;
	float	fDlt;
	float	fRate;
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex)
	{
		//最小値との差
		fDlt	= vecHghtPnt[nIndex] - fMinHght;
		//全幅に対する割合0-1
		fRate	= fDlt / fWdith;
		//レイヤーの番号
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
	UINT nLoopNum	= vecPartWdth.size();			//レイヤーの数
	vector<float> vecPartMax(nLoopNum, -FLT_MAX);	//各レイヤーにおける最大値
	vector<float> vecPartMin(nLoopNum, FLT_MAX);	//各レイヤーにおける最小値
	
	int nPart;
	for(UINT nIndex = 0; nIndex < pnt_size; ++nIndex){
		//レイヤーの番号
		nPart = vecPartNum[nIndex];
		//各レイヤーにおける最大・最小値
		if(vecPntCld[nIndex] > vecPartMax[nPart])
			vecPartMax[nPart] = vecPntCld[nIndex];
		if(vecPntCld[nIndex] < vecPartMin[nPart])
			vecPartMin[nPart] = vecPntCld[nIndex];
	}
	//各レイヤーの幅
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
	UINT			nLoopNum	= vecAvrgInt.size();	//レイヤーの数
	vector<UINT>	vecCount(nLoopNum, 0);				//有効な反射強度をもつポイントの数

	int nPart;
	for(UINT nIndex = 0; nIndex < pnt_size-1; ++nIndex){
		//有効な反射強度のみ使用
		if(vecIntValue[nIndex] > 0.0f){
			//レイヤーの番号
			nPart = vecPartNum[nIndex];

			vecAvrgInt[nPart] += vecIntValue[nIndex];
			++vecCount[nPart];
		}
	}
	//各レイヤーの平均反射強度
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

	//特徴ベクトルの作成
	int				tmp;
	UINT			nDimNum	= vecFtrList.size();	
	vector<float>	vecClFeature(nDimNum);			
	for(UINT nDim = 0; nDim < nDimNum; ++nDim){
		tmp = vecFtrList[nDim];
		vecClFeature[nDim] = vecFeature[tmp];
	}
	//特徴量を特徴強度に変換
	vector<vector<UINT>>	vecFtrStrngth;
	CreateFtrStrngth(vecClFeature, vecFtrStrngth, vecSplitPoint);

	//分類スコアの算出
	Prediction(vecFtrStrngth, vecWeakCl);
	vector<double>&	vecClScore = GetClScore();

	//分類スコアから確率に変換
	Score2Prob(vecClScore, vecSgmdParam);
}

inline	void	CObjClassify::CreateFtrStrngth(vector<float>& vecClFeature, vector<vector<UINT>>& vecFtrStrngth,  vector<vector<float>>& vecSplitPoint)
{
	//次元数
	UINT nDimNum	= vecSplitPoint.size();
	//メモリ確保
	vecFtrStrngth	= vector<vector<UINT>>(PROB_DIM, vector<UINT>(nDimNum));

	//二分探索より特徴強度を求める
	size_t tmp;
	vector<float>::iterator itrStrt;
	vector<float>::iterator itrBound;
	for(UINT i = 0; i < PROB_DIM; ++i)
	{
		for(UINT nDim = 0; nDim < nDimNum; ++nDim)
		{
			//BINの幅の一覧
			itrStrt		= vecSplitPoint[nDim].begin() + i*BIN[0];
			//二分探索
			itrBound	= upper_bound(itrStrt, itrStrt+BIN[i], vecClFeature[nDim]);
			//BINの番号
			tmp = distance(itrStrt, itrBound);
			if( (int)tmp > BIN[i] - 1)
				tmp = BIN[i] - 1;

			vecFtrStrngth[i][nDim] = tmp;
		}
	}
}

inline	void	CObjClassify::Prediction(vector<vector<UINT>>& vecFtrStrngth, vector<Classifier>& vecWeakCl)
{
	//学習回数
	UINT	nLearnNum = vecWeakCl.size() / CLASS_NUM;
	
	int				iDim;
	int				iNum;
	UINT			tmp;
	UINT			nBin[PROB_DIM];
	vector<double>	vecClScore(CLASS_NUM, 0.0);
	//学習回数ループ
	for(UINT nRound = 0; nRound < nLearnNum; ++nRound){
		//クラス数ループ
		for(UINT nClass = 0; nClass < CLASS_NUM; ++nClass){
			tmp	= nRound*CLASS_NUM + nClass;
			//特徴次元
			vector<int>& HistDim = vecWeakCl[tmp].HistDim;

			for(UINT i = 0; i < PROB_DIM; ++i){
				//次元
				iDim	= HistDim[i];
				//BINの番号
				nBin[i] = vecFtrStrngth[i][iDim];
			}
			iNum = nBin[1]*BIN[0] + nBin[0];

			//スコアの算出
			vecClScore[nClass] += vecWeakCl[tmp].Lut[iNum];
		}
	}

	m_vecClScore.swap(vecClScore);
}

inline	void	CObjClassify::Score2Prob(vector<double>& vecClScore, vector<vector<double>>& vecSgmdParam)
{
	//シグモイド関数より確率に変換
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

	//確率が最大のクラスへ識別する
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
	//特徴量計算
	if( !AnalyzeFeature(vecPntCld, sPose, cObj, cTrck) )
		return	FALSE;

	//識別-RealAdaBoost
	ClassifyRAB(GetFtrList(), GetWeakCl(), GetSplitPoint(), GetSigmoidParam());

	// クラス確率の保存
	vecProb	= GetClProb();

	return	TRUE;
}
