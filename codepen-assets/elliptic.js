// elliptic functions harvested from 
// https://github.com/paulmasson/math/tree/master/src/functions

function complex( x, y ) {

  var y = y || 0;
  return { re: x, im: y };

}

var C = complex;

function isComplex( x ) { return typeof x === 'object' && 're' in x }


function abs( x ) {

  if ( isComplex(x) ) {

    if ( x.re === 0 && x.im === 0 ) return 0;

    if ( Math.abs(x.re) < Math.abs(x.im) )

      return Math.abs(x.im) * Math.sqrt( 1 + ( x.re / x.im )**2 );

    else

      return Math.abs(x.re) * Math.sqrt( 1 + ( x.im / x.re )**2 );

  }

  return Math.abs(x);

}

function arg( x ) {

  if ( isComplex(x) ) return Math.atan2( x.im, x.re );

  return Math.atan2( 0, x );

}


// JavaScript does not support operator overloading

function add( x, y ) {

  if ( arguments.length > 2 ) {

    var z = add( x, y );
    for ( var i = 2 ; i < arguments.length ; i++ ) z = add( z, arguments[i] );
    return z; 

  }

  if ( isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    return { re: x.re + y.re, im: x.im + y.im };

  }

  return x + y;

}

function sub( x, y ) {

  if ( isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    return { re: x.re - y.re, im: x.im - y.im };

  }

  return x - y;

}

function mul( x, y ) {

  if ( arguments.length > 2 ) {

    var z = mul( x, y );
    for ( var i = 2 ; i < arguments.length ; i++ ) z = mul( z, arguments[i] );
    return z; 

  }

  if ( isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    return { re: x.re * y.re - x.im * y.im,
             im: x.im * y.re + x.re * y.im };

  }

  return x * y;

}

function neg( x ) { return mul( -1, x ); }

function div( x, y ) {

  // need to handle 0/0...

  if ( isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    if ( Math.abs(y.re) < Math.abs(y.im) ) {

      var f = y.re / y.im;
      return { re: ( x.re * f + x.im ) / ( y.re * f + y.im ),
               im: ( x.im * f - x.re ) / ( y.re * f + y.im ) };

    } else {

      var f = y.im / y.re;
      return { re: ( x.re + x.im * f ) / ( y.re + y.im * f ),
               im: ( x.im - x.re * f ) / ( y.re + y.im * f ) };

    }

  }

  return x / y;

}

function inv( x ) { return div( 1, x ); }

function pow( x, y ) {

  if ( isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    if ( x.re === 0 && x.im === 0 && y.re > 0 )
      return complex(0);
    if ( x.re === 0 && x.im === 0 && y.re === 0 && y.im === 0 )
      return complex(1);
    if ( x.re === 0 && x.im === 0 && y.re < 0 )
      throw 'Power singularity';

    var r = Math.sqrt( x.re * x.re + x.im * x.im );
    var phi = Math.atan2( x.im, x.re );

    var R = r**y.re * Math.exp( -phi * y.im );
    var Phi = phi * y.re + y.im * Math.log(r);

    return { re: R * Math.cos(Phi), im: R * Math.sin(Phi) };

  }

  if ( x < 0 ) return pow( complex(x), y );

  return x**y;

}

function root( x, y ) { return pow( x, div( 1, y ) ); }

function sqrt( x ) {

  if ( isComplex(x) ) {

    var R = ( x.re * x.re + x.im * x.im )**(1/4);
    var Phi = Math.atan2( x.im, x.re ) / 2;

    return { re: R * Math.cos(Phi), im: R * Math.sin(Phi) };

  }

  if ( x < 0 ) return sqrt( complex(x) );

  return Math.sqrt(x);

}

function jacobiTheta( n, x, q, tolerance=1e-10 ) {

  if ( abs(q) >= 1 ) throw 'Unsupported elliptic nome';

  if ( ![1,2,3,4].includes(n) ) throw 'Undefined Jacobi theta index';

  if ( isComplex(x) || isComplex(q) ) {

    if ( !isComplex(x) ) x = complex(x);

    var piTau = div( log(q), complex(0,1) );

    // dlmf.nist.gov/20.2 to reduce overflow
    if ( Math.abs(x.im) > Math.abs(piTau.im) || Math.abs(x.re) > Math.PI ) {

      var pt = Math.round( x.im / piTau.im );
      x = sub( x, mul( pt, piTau ) );

      var p = Math.round( x.re / Math.PI );
      x = sub( x, p * Math.PI );

      var qFactor = pow( q, -pt*pt );
      var eFactor = exp( mul( -2 * pt, x, complex(0,1) ) );

      // factors can become huge, so chop spurious parts first
      switch( n ) {

        case 1:

          return mul( (-1)**(p+pt), qFactor, eFactor, chop( jacobiTheta( n, x, q ), tolerance ) );

        case 2:

          return mul( (-1)**p, qFactor, eFactor, chop( jacobiTheta( n, x, q ), tolerance ) );

        case 3:

          return mul( qFactor, eFactor, chop( jacobiTheta( n, x, q ), tolerance ) );

        case 4:

          return mul( (-1)**pt, qFactor, eFactor, chop( jacobiTheta( n, x, q ), tolerance ) );

      }

    }

    switch( n ) {

      case 1:

        var s = complex(0);
        var p = complex(1);
        var i = 0;

        while ( Math.abs(p.re) > tolerance || Math.abs(p.im) > tolerance ) {
          p = mul( (-1)**i, pow( q, i*i+i ), sin( mul(2*i+1,x) ) );
          s = add( s, p );
          i++;
        }

        return mul( 2, pow( q, 1/4 ), s );

      case 2:

        var s = complex(0);
        var p = complex(1);
        var i = 0;

        while ( Math.abs(p.re) > tolerance || Math.abs(p.im) > tolerance ) {
          p = mul( pow( q, i*i+i ), cos( mul(2*i+1,x) ) );
          s = add( s, p );
          i++;
        }

        return mul( 2, pow( q, 1/4 ), s );

      case 3:

        var s = complex(0);
        var p = complex(1);
        var i = 1;

        while ( Math.abs(p.re) > tolerance || Math.abs(p.im) > tolerance ) {
          p = mul( pow( q, i*i ), cos( mul(2*i,x) ) );
          s = add( s, p );
          i++;
        }

        return add( 1, mul(2,s) );

      case 4:

        var s = complex(0);
        var p = complex(1);
        var i = 1;

        while ( Math.abs(p.re) > tolerance || Math.abs(p.im) > tolerance ) {
          p = mul( pow( neg(q), i*i ), cos( mul(2*i,x) ) );
          s = add( s, p );
          i++;
        }

        return add( 1, mul(2,s) );

      }

  }

  // dlmf.nist.gov/20.2 to reduce overflow
  if ( Math.abs(x.re) > Math.PI ) {

    var p = Math.round( x / Math.PI );
    x = x - p * Math.PI;

    switch( n ) {

      case 1:
      case 2:

        return (-1)**p * jacobiTheta( n, x, q );

      case 3:
      case 4:

        return jacobiTheta( n, x, q );

    }

  }

  switch( n ) {

    case 1:

      if ( q < 0 ) return jacobiTheta( n, x, complex(q) );

      var s = 0;
      var p = 1;
      var i = 0;

      while ( Math.abs(p) > tolerance ) {
        p = (-1)**i * q**(i*i+i) * sin( (2*i+1) * x );
        s += p;
        i++;
      }

      return 2 * q**(1/4) * s;

    case 2:

      if ( q < 0 ) return jacobiTheta( n, x, complex(q) );

      var s = 0;
      var p = 1;
      var i = 0;

      while ( Math.abs(p) > tolerance ) {
        p = q**(i*i+i) * cos( (2*i+1) * x );
        s += p;
        i++;
      }

      return 2 * q**(1/4) * s;

    case 3:

      var s = 0;
      var p = 1;
      var i = 1;

      while ( Math.abs(p) > tolerance ) {
        p = q**(i*i) * cos( 2*i * x );
        s += p;
        i++;
      }

      return 1 + 2 * s;

    case 4:

      var s = 0;
      var p = 1;
      var i = 1;

      while ( Math.abs(p) > tolerance ) {
        p = (-q)**(i*i) * cos( 2*i * x );
        s += p;
        i++;
      }

      return 1 + 2 * s;

  }

}


function ellipticNome( m ) {

  if ( isComplex(m) ) return exp( div( mul( -pi, ellipticK( sub(1,m) ) ), ellipticK(m) ) );

  if ( m > 1 ) return ellipticNome( complex(m) );

  if ( m < 0 ) return -exp( -pi * ellipticK( 1/(1-m) ) / ellipticK( m/(m-1) ) );

  return exp( -pi * ellipticK(1-m) / ellipticK(m) );

}


function sn( x, m ) {

  var q = ellipticNome(m);

  if ( m > 1 || isComplex(x) || isComplex(m) ) {

    var t = div( x, pow( jacobiTheta(3,0,q), 2 ) );

    return mul( div( jacobiTheta(3,0,q), jacobiTheta(2,0,q) ),
                div( jacobiTheta(1,t,q), jacobiTheta(4,t,q) ) );

  }

  var t = x / jacobiTheta(3,0,q)**2;

  if ( m < 0 )
    return jacobiTheta(3,0,q) / jacobiTheta(4,t,q)
           * div( jacobiTheta(1,t,q), jacobiTheta(2,0,q) ).re;

  return jacobiTheta(3,0,q) / jacobiTheta(2,0,q)
         * jacobiTheta(1,t,q) / jacobiTheta(4,t,q);

}

function cn( x, m ) {

  var q = ellipticNome(m);

  if ( m > 1 || isComplex(x) || isComplex(m) ) {

    var t = div( x, pow( jacobiTheta(3,0,q), 2 ) );

    return mul( div( jacobiTheta(4,0,q), jacobiTheta(2,0,q) ),
                div( jacobiTheta(2,t,q), jacobiTheta(4,t,q) ) );

  }

  var t = x / jacobiTheta(3,0,q)**2;

  if ( m < 0 )
    return jacobiTheta(4,0,q) / jacobiTheta(4,t,q)
           * div( jacobiTheta(2,t,q), jacobiTheta(2,0,q) ).re;

  return jacobiTheta(4,0,q) / jacobiTheta(2,0,q)
         * jacobiTheta(2,t,q) / jacobiTheta(4,t,q);

}

function dn( x, m ) {

  var q = ellipticNome(m);

  if ( m > 1 || isComplex(x) || isComplex(m) ) {

    var t = div( x, pow( jacobiTheta(3,0,q), 2 ) );

    return mul( div( jacobiTheta(4,0,q), jacobiTheta(3,0,q) ),
                div( jacobiTheta(3,t,q), jacobiTheta(4,t,q) ) );

  }

  var t = x / jacobiTheta(3,0,q)**2;

  return jacobiTheta(4,0,q) / jacobiTheta(3,0,q)
         * jacobiTheta(3,t,q) / jacobiTheta(4,t,q);

}

function am( x, m ) {

  if ( m > 1 || isComplex(x) || isComplex(m) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(m) ) m = complex(m);

    if ( m.im === 0 && m.re <= 1 ) {

      var K = ellipticK( m.re );
      var n = Math.round( x.re / 2 / K );
      x = sub( x, 2 * n * K );

      if ( m.re < 0 ) {

        var Kp = ellipticK( 1 - m.re );
        var p = Math.round( x.im / 2 / Kp.re );

        // bitwise test for odd integer
        if ( p & 1 ) return sub( n * pi, arcsin( sn(x,m) ) );

      }

      return add( arcsin( sn(x,m) ), n * pi );

    }

    return arcsin( sn(x,m) );

  } else {

    var K = ellipticK(m);
    var n = Math.round( x / 2 / K );
    x = x - 2 * n * K;

    return Math.asin( sn(x,m) ) + n * pi;

  }

}


function weierstrassP( x, g2, g3 ) {

  if ( !isComplex(x) ) x = complex(x);

  function cubicTrigSolution( p, q, n ) {

    // p, q both negative in defining cubic

    return mul( 2/sqrt(3), sqrt(p),
                cos( sub( div( arccos( mul( 3*sqrt(3)/2, q, pow(p,-3/2) ) ), 3 ),
                          2*pi*n/3 ) ) );
  }

  g2 = div( g2, 4 );
  g3 = div( g3, 4 );

  var e1 = cubicTrigSolution( g2, g3, 0 );
  var e2 = cubicTrigSolution( g2, g3, 1 );
  var e3 = cubicTrigSolution( g2, g3, 2 );

  // Whittaker & Watson, Section 22.351

  var m = div( sub(e2,e3), sub(e1,e3) );

  return add( e3, mul( sub(e1,e3), pow( sn( mul( x, sqrt(sub(e1,e3)) ), m ), -2 ) ) );

}

function exp( x ) {

  if ( isComplex(x) )

    return { re: Math.exp(x.re) * Math.cos(x.im),
             im: Math.exp(x.re) * Math.sin(x.im) };

  return Math.exp(x);

}


function log( x, base ) {

  if ( isComplex(x) ) {

    if ( isComplex(base) ) return div( log(x), log(base) );

    return { re: log( abs(x), base ), im: log( Math.E, base ) * arg(x) };

  }

  if ( x < 0 ) return log( complex(x), base );

  if ( base === undefined ) return Math.log(x);

  return Math.log(x) / Math.log(base);

}

var ln = log;


function lambertW( k, x ) {

  if ( arguments.length === 1 ) {
    x = k;
    k = 0;
  }

  if ( Math.abs( x + Math.exp(-1) ) < 1e-16 ) return -1;

  // inversion by root finding

  switch ( k ) {

    case 0:

      if ( x < -Math.exp(-1) ) throw 'Unsupported lambertW argument';

      return findRoot( w => w * Math.exp(w) - x, [-1,1000], { tolerance: 1e-16 } );

    case -1:

      if ( x < -Math.exp(-1) || x > 0 ) throw 'Unsupported lambertW argument';

      return findRoot( w => w * Math.exp(w) - x, [-1000,-1], { tolerance: 1e-16 } );

    default:

      throw 'Unsupported lambertW index';

  }

}
var pi = Math.PI;

// Carlson symmetric integrals

function carlsonRC( x, y ) {

  if ( x < 0 || y < 0 || isComplex(x) || isComplex(y) ) {

    if ( !isComplex(x) ) x = complex(x);
    if ( !isComplex(y) ) y = complex(y);

    if ( x.re === y.re && x.im === y.im ) return inv( sqrt(x) );

    return div( arccos( sqrt( div(x,y) ) ), sqrt( sub(y,x) ) );

  }

  if ( x === y ) return 1 / Math.sqrt(x);

  if ( x < y )
    return Math.acos( Math.sqrt(x/y) ) / Math.sqrt(y-x);
  else
    return Math.acosh( Math.sqrt(x/y) ) / Math.sqrt(x-y);

}

function carlsonRD( x, y, z ) {

  return carlsonRJ( x, y, z, z );

}

function carlsonRF( x, y, z, tolerance=1e-10 ) {

  if ( isComplex(x) || isComplex(y) || isComplex(z) ) {

    var xm = x;
    var ym = y;
    var zm = z;

    var Am = A0 = div( add( x, y, z ), 3 );
    var Q = Math.pow( 3*tolerance, -1/6 )
            * Math.max( abs( sub(A0,x) ), abs( sub(A0,y) ), abs( sub(A0,z) ) );
    var g = .25;
    var pow4 = 1;
    var m = 0;

    while ( true ) {
      var xs = sqrt(xm);
      var ys = sqrt(ym);
      var zs = sqrt(zm);
      var lm = add( mul(xs,ys), mul(xs,zs), mul(ys,zs) );
      var Am1 = mul( add(Am,lm), g );
      xm = mul( add(xm,lm), g );
      ym = mul( add(ym,lm), g );
      zm = mul( add(zm,lm), g );
      if ( pow4 * Q < abs(Am) ) break;
      Am = Am1;
      m += 1;
      pow4 *= g;
    }

    var t = div( pow4, Am );
    var X = mul( sub(A0,x), t );
    var Y = mul( sub(A0,y), t );
    var Z = neg( add(X,Y) );
    var E2 = sub( mul(X,Y), mul(Z,Z) );
    var E3 = mul(X,Y,Z);

    return mul( pow( Am, -.5 ),
             add( 9240, mul(-924,E2), mul(385,E2,E2), mul(660,E3), mul(-630,E2,E3) ), 1/9240 );

  } else {

    if ( y === z ) return carlsonRC( x, y );
    if ( x === z ) return carlsonRC( y, x );
    if ( x === y ) return carlsonRC( z, x );

    // adapted from mpmath / elliptic.py

    var xm = x;
    var ym = y;
    var zm = z;

    var Am = A0 = (x + y + z) / 3;
    var Q = Math.pow( 3*tolerance, -1/6 )
            * Math.max( Math.abs(A0-x), Math.abs(A0-y), Math.abs(A0-z) );
    var g = .25;
    var pow4 = 1;
    var m = 0;

    while ( true ) {
      var xs = Math.sqrt(xm);
      var ys = Math.sqrt(ym);
      var zs = Math.sqrt(zm);
      var lm = xs*ys + xs*zs + ys*zs;
      var Am1 = (Am + lm) * g;
      xm = (xm + lm) * g;
      ym = (ym + lm) * g;
      zm = (zm + lm) * g;
      if ( pow4 * Q < Math.abs(Am) ) break;
      Am = Am1;
      m += 1;
      pow4 *= g;
    }

    var t = pow4 / Am;
    var X = (A0-x) * t;
    var Y = (A0-y) * t;
    var Z = -X-Y;
    var E2 = X*Y - Z**2;
    var E3 = X*Y*Z;

    return Math.pow( Am, -.5 )
           * ( 9240 - 924*E2 + 385*E2**2 + 660*E3 - 630*E2*E3 ) / 9240;

  }

}

function carlsonRG( x, y, z ) {

  return 1;

}

function carlsonRJ( x, y, z, p, tolerance=1e-10 ) {

  if ( isComplex(x) || isComplex(y) || isComplex(z) || isComplex(p) ) {

    var xm = x;
    var ym = y;
    var zm = z;
    var pm = p;

    var A0 = Am = div( add( x, y, z, mul(2,p) ), 5 );
    var delta = mul( sub(p,x), sub(p,y), sub(p,z) );
    var Q = Math.pow( .25*tolerance, -1/6 )
            * Math.max( abs( sub(A0,x) ), abs( sub(A0,y) ), abs( sub(A0,z) ), abs( sub(A0,p) ) );
    var m = 0;
    var g = .25;
    var pow4 = 1;
    var S = complex(0);

    while ( true ) {
      var sx = sqrt(xm);
      var sy = sqrt(ym);
      var sz = sqrt(zm);
      var sp = sqrt(pm);
      var lm = add( mul(sx,sy), mul(sx,sz), mul(sy,sz) );
      var Am1 = mul( add(Am,lm), g );
      xm = mul( add(xm,lm), g );
      ym = mul( add(ym,lm), g );
      zm = mul( add(zm,lm), g );
      pm = mul( add(pm,lm), g );
      var dm = mul( add(sp,sx), add(sp,sy), add(sp,sz) );
      var em = mul( delta, Math.pow( 4, -3*m ), inv(dm), inv(dm) );
      if ( pow4 * Q < abs(Am) ) break;
      var T = mul( carlsonRC( 1, add(1,em) ), pow4, inv(dm) );
      S = add( S, T );
      pow4 *= g;
      m += 1;
      Am = Am1;
    }

    var t = div( Math.pow( 2, -2*m ), Am );
    var X = mul( sub(A0,x), t );
    var Y = mul( sub(A0,y), t );
    var Z = mul( sub(A0,z), t );
    var P = div( add(X,Y,Z), -2 );
    var E2 = add( mul(X,Y), mul(X,Z), mul(Y,Z), mul(-3,P,P) );
    var E3 = add( mul(X,Y,Z), mul(2,E2,P), mul(4,P,P,P) );
    var E4 = mul( add( mul(2,X,Y,Z), mul(E2,P), mul(3,P,P,P) ), P );
    var E5 = mul(X,Y,Z,P,P);
    P = add( 24024, mul(-5148,E2), mul(2457,E2,E2), mul(4004,E3), mul(-4158,E2,E3), mul(-3276,E4), mul(2772,E5) );
    var v1 = mul( g**m, pow( Am, -1.5 ), P, 1/24024 );
    var v2 = mul(6,S);

    return add( v1, v2 );

  } else {

    // adapted from mpmath / elliptic.py

    var xm = x;
    var ym = y;
    var zm = z;
    var pm = p;

    var A0 = Am = (x + y + z + 2*p) / 5;
    var delta = (p-x) * (p-y) * (p-z);
    var Q = Math.pow( .25*tolerance, -1/6 )
            * Math.max( Math.abs(A0-x), Math.abs(A0-y), Math.abs(A0-z), Math.abs(A0-p) );
    var m = 0;
    var g = .25;
    var pow4 = 1;
    var S = 0;

    while ( true ) {
      var sx = Math.sqrt(xm);
      var sy = Math.sqrt(ym);
      var sz = Math.sqrt(zm);
      var sp = Math.sqrt(pm);
      var lm = sx*sy + sx*sz + sy*sz;
      var Am1 = (Am + lm) * g;
      xm = (xm + lm) * g;
      ym = (ym + lm) * g;
      zm = (zm + lm) * g;
      pm = (pm + lm) * g;
      var dm = (sp+sx) * (sp+sy) * (sp+sz);
      var em = delta * Math.pow( 4, -3*m ) / dm**2;
      if ( pow4 * Q < Math.abs(Am) ) break;
      var T = carlsonRC( 1, 1 + em ) * pow4 / dm;
      S += T;
      pow4 *= g;
      m += 1;
      Am = Am1;
    }

    var t = Math.pow( 2, -2*m ) / Am;
    var X = (A0-x) * t;
    var Y = (A0-y) * t;
    var Z = (A0-z) * t;
    var P = (-X-Y-Z) / 2;
    var E2 = X*Y + X*Z + Y*Z - 3*P**2;
    var E3 = X*Y*Z + 2*E2*P + 4*P**3;
    var E4 = ( 2*X*Y*Z + E2*P + 3*P**3 ) * P;
    var E5 = X*Y*Z*P**2;
    P = 24024 - 5148*E2 + 2457*E2**2 + 4004*E3 - 4158*E2*E3 - 3276*E4 + 2772*E5;
    var v1 = g**m * Math.pow( Am, -1.5 ) * P / 24024;
    var v2 = 6*S;

    return v1 + v2;

  }

}


// elliptic integrals

function ellipticF( x, m ) {

  if ( arguments.length === 1 ) {
    m = x;
    x = pi / 2;
  }

  if ( isComplex(x) || isComplex(m) ) {

    if ( !isComplex(x) ) x = complex(x);

    var period = complex(0);
    if ( Math.abs(x.re) > pi / 2 ) {
      var p = Math.round( x.re / pi );
      x.re = x.re - p * pi;
      period = mul( 2 * p, ellipticK( m ) );
    }

    return add( mul( sin(x), carlsonRF( mul(cos(x),cos(x)), sub( 1, mul(m,sin(x),sin(x)) ), 1 ) ), period );

  } else {

    if ( m > 1 && x > Math.asin( 1 / Math.sqrt(m) ) ) return ellipticF( complex(x), m );

    var period = 0;
    if ( Math.abs(x) > pi / 2 ) {
      var p = Math.round( x / pi );
      x = x - p * pi;
      period = 2 * p * ellipticK( m );
    }

    return sin(x) * carlsonRF( cos(x)**2, 1 - m * sin(x)**2, 1 ) + period;

  }

}

function ellipticK( m ) {

  return ellipticF( m );

}

function ellipticE( x, m ) {

  if ( arguments.length === 1 ) {
    m = x;
    x = pi / 2;
  }

  if ( isComplex(x) || isComplex(m) ) {

    if ( !isComplex(x) ) x = complex(x);

    var period = complex(0);
    if ( Math.abs(x.re) > pi / 2 ) {
      var p = Math.round( x.re / pi );
      x.re = x.re - p * pi;
      period = mul( 2 * p,  ellipticE( m ) );
    }

    return add( mul( sin(x), carlsonRF( mul(cos(x),cos(x)), sub( 1, mul(m,sin(x),sin(x)) ), 1 ) ),
                mul( -1/3, m, pow(sin(x),3), carlsonRD( mul(cos(x),cos(x)), sub( 1, mul(m,sin(x),sin(x)) ), 1 ) ),
                period );

  } else {

    if ( m > 1 && x > Math.asin( 1 / Math.sqrt(m) ) ) return ellipticE( complex(x), m );

    var period = 0;
    if ( Math.abs(x) > pi / 2 ) {
      var p = Math.round( x / pi );
      x = x - p * pi;
      period = 2 * p * ellipticE( m );
    }

    return sin(x) * carlsonRF( cos(x)**2, 1 - m * sin(x)**2, 1 )
           - m / 3 * sin(x)**3 * carlsonRD( cos(x)**2, 1 - m * sin(x)**2, 1 )
           + period;

  }

}

function ellipticPi( n, x, m ) {

  if ( arguments.length === 2 ) {
    m = x;
    x = pi / 2;
  }

  if ( isComplex(n) || isComplex(x) || isComplex(m) ) {

    if ( !isComplex(x) ) x = complex(x);

    var period = complex(0);
    if ( Math.abs(x.re) > pi / 2 ) {
      var p = Math.round( x.re / pi );
      x.re = x.re - p * pi;
      period = mul( 2 * p, ellipticPi( n, m ) );
    }

    return add( mul( sin(x), carlsonRF( mul(cos(x),cos(x)), sub( 1, mul(m,sin(x),sin(x)) ), 1 ) ),
                mul( 1/3, n, pow(sin(x),3),
                  carlsonRJ( mul(cos(x),cos(x)), sub( 1, mul(m,sin(x),sin(x)) ), 1, sub( 1, mul(n,sin(x),sin(x)) ) ) ),
                period );

  } else {

    if ( n > 1 && x > Math.asin( 1 / Math.sqrt(n) ) ) return ellipticPi( n, complex(x), m );

    if ( m > 1 && x > Math.asin( 1 / Math.sqrt(m) ) ) return ellipticPi( n, complex(x), m );

    var period = 0;
    if ( Math.abs(x) > pi / 2 ) {
      var p = Math.round( x / pi );
      x = x - p * pi;
      period = 2 * p * ellipticPi( n, m );
    }

    return sin(x) * carlsonRF( cos(x)**2, 1 - m * sin(x)**2, 1 )
           + n / 3 * sin(x)**3
             * carlsonRJ( cos(x)**2, 1 - m * sin(x)**2, 1, 1 - n * sin(x)**2 )
           + period;

  }

}


function jacobiZeta( x, m ) {

  // using definition matching elliptic integrals
  // alternate definition replaces x with am(x,m)

  return sub( ellipticE( x, m ), mul( ellipticF(x,m), ellipticE(m), inv( ellipticK(m) ) ) );

}

// complex circular functions

function sin( x ) {

  if ( isComplex(x) )

    return { re: Math.sin(x.re) * Math.cosh(x.im),
             im: Math.cos(x.re) * Math.sinh(x.im) };

  return Math.sin(x);

}

function cos( x ) {

  if ( isComplex(x) )

    return { re: Math.cos(x.re) * Math.cosh(x.im),
             im: -Math.sin(x.re) * Math.sinh(x.im) };

  return Math.cos(x);

}

function tan( x ) {

  if ( isComplex(x) ) return div( sin(x), cos(x) );

  return Math.tan(x);

 }

function cot( x ) {

  if ( isComplex(x) ) return div( cos(x), sin(x) );

  return 1 / Math.tan(x);

}

function sec( x ) {

  if ( isComplex(x) ) return div( 1, cos(x) );

  return 1 / Math.cos(x);

}

function csc( x ) {

  if ( isComplex(x) ) return div( 1, sin(x) );

  return 1 / Math.sin(x);

}


// inverse circular functions

function arcsin( x ) {

  if ( isComplex(x) ) {

    var s = sqrt( sub( 1, mul( x, x ) ) );
    s = add( mul( complex(0,1), x ), s ); 
    return mul( complex(0,-1), log( s ) );

  }

  if ( Math.abs(x) <= 1 ) return Math.asin(x);

  return arcsin( complex(x) );

}

function arccos( x ) {

  if ( isComplex(x) ) {

    return sub( pi/2, arcsin(x) );

  }

  if ( Math.abs(x) <= 1 ) return Math.acos(x);

  return arccos( complex(x) );

}

function arctan( x ) {

  if ( isComplex(x) ) {

    var s = sub( log( sub( 1, mul( complex(0,1), x ) ) ),
                 log( add( 1, mul( complex(0,1), x ) ) ) );
    return mul( complex(0,1/2), s );

  }

  return Math.atan(x);

}

function arccot( x ) {

  if ( isComplex(x) ) return arctan( div( 1, x ) );

  return Math.atan( 1/x );

}

function arcsec( x ) {

  if ( isComplex(x) ) return arccos( div( 1, x ) );

  if ( Math.abs(x) >= 1 ) return Math.acos( 1/x );

  return arcsec( complex(x) );

}

function arccsc( x ) {

  if ( isComplex(x) ) return arcsin( div( 1, x ) );

  if ( Math.abs(x) >= 1 ) return Math.asin( 1/x );

  return arccsc( complex(x) );

}


// complex hyperbolic functions

function sinh( x ) {

  if ( isComplex(x) )

    return { re: Math.sinh(x.re) * Math.cos(x.im),
             im: Math.cosh(x.re) * Math.sin(x.im) };

  return Math.sinh(x);

}

function cosh( x ) {

  if ( isComplex(x) )

    return { re: Math.cosh(x.re) * Math.cos(x.im),
             im: Math.sinh(x.re) * Math.sin(x.im) };

  return Math.cosh(x);

}

function tanh( x ) {

  if ( isComplex(x) ) return div( sinh(x), cosh(x) );

  return Math.tanh(x);

}

function coth( x ) {

  if ( isComplex(x) ) return div( cosh(x), sinh(x) );

  return 1 / Math.tanh(x);

}

function sech( x ) {

  if ( isComplex(x) ) return div( 1, cosh(x) );

  return 1 / Math.cosh(x);

}

function csch( x ) {

  if ( isComplex(x) ) return div( 1, sinh(x) );

  return 1 / Math.sinh(x);

}


// inverse hyperbolic functions

function arcsinh( x ) {

  if ( isComplex(x) ) {

    var s = sqrt( add( mul( x, x ), 1 ) );
    s = add( x, s );
    return log( s );

  }

  return Math.asinh(x);

}

function arccosh( x ) {

  if ( isComplex(x) ) {

    var s = mul( sqrt( add( x, 1 ) ), sqrt( sub( x, 1 ) ) );
    s = add( x, s ); 
    return log( s );

  }

  if ( x >= 1 ) return Math.acosh(x);

  return arccosh( complex(x) );

}

function arctanh( x ) {

  if ( isComplex(x) ) {

    var s = sub( log( add( 1, x ) ), log( sub( 1, x ) ) );
    return mul( 1/2, s );

  }

  if ( Math.abs(x) <= 1 ) return Math.atanh(x);

  return arctanh( complex(x) );

}

function arccoth( x ) {

  if ( isComplex(x) ) {

    if ( x.re === 0 && x.im === 0 ) throw 'Indeterminate value';

    return arctanh( div( 1, x ) );

  }

  if ( Math.abs(x) > 1 ) return Math.atanh( 1/x );

  return arccoth( complex(x) );

}

function arcsech( x ) {

  if ( isComplex(x) ) {

    if ( x.re === 0 && x.im === 0 ) throw 'Indeterminate value';

    // adjust for branch cut along negative axis
    if ( x.im === 0 ) x.im = -1e-300;

    return arccosh( div( 1, x ) );

  }

  if ( x > 0 && x < 1 ) return Math.acosh( 1/x );

  return arcsech( complex(x) );

}

function arccsch( x ) {

  if ( isComplex(x) ) {

    return arcsinh( div( 1, x ) );

  }

  return Math.asinh( 1/x );

}


// miscellaneous

function sinc( x ) {

  if ( isComplex(x) ) {

    if ( x.re === 0 && x.im === 0 ) return complex(1);

    return div( sin(x), x );

  }

  if ( x === 0 ) return 1;

  return Math.sin(x) / x;

}

function chop( x, tolerance=1e-10 ) {

  if ( Array.isArray(x) ) {
    var v = vector( x.length );
    for ( var i = 0 ; i < x.length ; i++ ) v[i] = chop( x[i] );
    return v;
  }

  if ( isComplex(x) ) return { re: chop(x.re), im: chop(x.im) };

  if ( Math.abs(x) < tolerance ) x = 0;

  return x;

}


function kronecker( i, j ) {

  return i === j ? 1 : 0;

}

