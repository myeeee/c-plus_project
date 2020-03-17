//////////////////////////////////////////////////////////////////////////////////////////////////
// BaseDefine.h
#ifndef	_BASEDEFINE_H_
#define	_BASEDEFINE_H_

#define	VST_PAI			(3.1415926535897932384626433832795)

#ifndef	PAI
	#define	PAI			VST_PAI
#endif	// POW_SQ

#ifndef	POW_SQ
	#define	POW_SQ(x)	((x)*(x))
#endif	// POW_SQ

#ifndef	POW_CB
	#define	POW_CB(x)	((x)*(x)*(x))
#endif	// POW_CB

#ifndef	Deg2Rad
	#define	Deg2Rad(X)	((X)*3.1415926535897932384626433832795/180)
#endif	// Deg2Rad

#ifndef	Rad2Deg
	#define	Rad2Deg(X)	((X)*180/3.1415926535897932384626433832795)
#endif	// Rad2Deg


#define	LIMIT(l,x,h)		( max((l), min((x),(h))) )

#include	<float.h>
#include	<math.h>
#include	<windows.h>
#include	"PosDef.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
inline	double	g_rand( double myu, double sigma )
{
	static BOOL	bIsFirstCall	= TRUE;

	if( bIsFirstCall )
	{
		srand( (unsigned int)time(NULL) );
		bIsFirstCall	= FALSE;
	}

	double	r1	= (double)(1+rand())/(1+RAND_MAX);
	double	r2	= (double)rand()/RAND_MAX;
	
	return	sigma * sqrt(-2.0*log(r1)) * sin( 2.0*PAI*r2 ) + myu; 
}

inline	double	round( double value )
{
	// 浮動小数点の四捨五入を行う
//	if( value>0 )	return	floor( 0.5+value );
//	else			return	-floor( 0.5-value );
	int	nValue;
	if( value>0 )	nValue	= (int)( value+0.5 );
	else			nValue	= (int)( value-0.5 );
	return	(double)nValue;
}

inline	float	round( float value )
{
	// 浮動小数点の四捨五入を行う
//	if( value>0 )	return	floor( 0.5f+value );
//	else			return	-floor( 0.5f-value );
	int	nValue;
	if( value>0 )	nValue	= (int)( value+0.5f );
	else			nValue	= (int)( value-0.5f );
	return	(float)nValue;
}

inline	double	NrmAngle( double value )
{
	// 任意の角度[rad] を ±π[rad] に制限
	value	= fmod( value, 2*PAI );
	return	( fabs(value)<PAI )	? value	: value-_copysign(2*PAI,value);
}

#endif		// _BASEDEFINE_H_