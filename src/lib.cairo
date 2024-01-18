// Struct representing float numbers using sign, mantissa and exponent.
// When Cairo language gets the update to support signed integers, the sign field will be removed
#[derive(Copy, Drop)]
struct Float {
    sign: u8,
    mantissa: u256,
    exponent: u256,
}

// Float number precision
const precision : u256 = 7;

// Computes the power of a given base raised to the specified exponent
fn pow (mut base: u256, mut exp: u256) -> u256 {
    let mut res = 1;
    loop {
        if exp == 0 {
            break();
        }
        res = res * base;
        exp -= 1;
    };

    res
}

// ReLU activation function used for neural network ML models
fn relu(x: Float) -> Float {
    let mut res = x;
    if x.sign == 1 {
        res = Float { sign: 0, mantissa: 0, exponent: 100 };
    } 

    res
}

// Truncate Float to "precision" number of digits, 5 in the example
fn truncate(num: Float) -> Float {

    let maxValue : u256 = pow(10, precision);
    let mut decValue : u256 = 1;
    let mut logValue : u256 = 0;

    loop {
        if num.mantissa < decValue {
            break();
        }  
        decValue *= 10; 
        logValue += 1;
    };

    let mut res : Float = Float { sign: num.sign, mantissa: num.mantissa, exponent: num.exponent };

    if logValue > precision {
        let diff = decValue / maxValue;
        res = Float { sign: num.sign, mantissa: num.mantissa / diff, exponent: num.exponent + (logValue - precision)};  // 
    }

    if res.mantissa == 0 {
        res = Float { sign: res.sign, mantissa: 0, exponent: 100 };
    }
    
    res
}

// Multiplication of Float numbers
fn mulFloats(x: Float, y: Float) -> Float {
    let m = x.mantissa * y.mantissa;
    let e = x.exponent + y.exponent - 100_u256;

    let sign = if x.sign != y.sign {
        1
    } else {
        0
    };

    truncate(Float { sign: sign, mantissa: m, exponent: e })
}

// Dividing of Float numbers
fn divFloats(x: Float, y: Float) -> Float {

    assert(y.mantissa > 0, 'Cannot divide by zero');

    let mut exp1: u256 = x.exponent;
    let mut mant1: u256 = x.mantissa;
    
    let exp2: u256 = y.exponent;
    let mant2: u256 = y.mantissa;

    // Can't divide lower by higher number with same precision, result will be 0
    // The lower must be multiplied by 10, it means at the same time exponent must be reduced by 1
    if mant1 < mant2 {
        mant1 *= 10; 
        exp1 -= 1;
    }

    let mut new_mant: u256 = 0;
    let mut i = 0;

    loop {
        if i == precision {
            break();
        }

        let div = mant1 / mant2;
        mant1 = (mant1 - mant2 * div) * 10; 
        
        // For precision N, the highest exponent is 10^(N-1)
        let exp = precision - i - 1;
        let pow = pow(10, exp);
        new_mant += div * pow;
        i += 1;
    };

    let new_exp = 100 + exp1 - exp2 - precision + 1;

    let new_sign = if x.sign != y.sign {
        1
    } else {
        0
    };

    Float{ sign: new_sign, mantissa: new_mant, exponent: new_exp }
}

// Sumation of Float numbers
fn addFloats(x: Float, y: Float) -> Float {
    let mut mant_1 = x.mantissa;
    let mut mant_2 = y.mantissa;

    let mut exp_1 = x.exponent;
    let mut exp_2 = y.exponent;

    let mut diff = 0;

    if exp_1 > exp_2 {
        diff = exp_1 - exp_2;
    } else {
        diff = exp_2 - exp_1;
    }

    let pow10 = pow(10, diff);

    if x.exponent < y.exponent {
        mant_2 *= pow10;
        exp_1 = x.exponent;
    } else {
        mant_1 *= pow10;
        exp_1 = y.exponent;
    }

    let mut sum_mant = mant_1 + mant_2;
    let mut sign = x.sign;

    if x.sign != y.sign {
        if mant_1 > mant_2 {
            sum_mant = mant_1 - mant_2;
        } else {
            sum_mant = mant_2 - mant_1;
            sign = y.sign;
        }
    }

    truncate(Float { sign: sign, mantissa: sum_mant, exponent: exp_1 })
}

// Subtraction of Float numbers
fn subFloats(x : Float, y : Float) -> Float {
    addFloats(x, Float { sign: 1 - y.sign, mantissa: y.mantissa, exponent: y.exponent })
}
// A_1_0_0 = 5100000e94 * 679225e86 = 3464047e86
// A_1_0_1 = 3500000e94 * -3933992e90 = -1376897e91
// A_1_0_2 = 1400000e94 * -3088550e89 = -4323970e89
// A_1_0_3 = 2000000e93 * 0e100 = 0e100
// B_0_0_0 = 3464047e86 + -1376897e91 = -1376862e91
// B_0_0_1 = -1376862e91 + -4323970e89 = -1420101e91
// B_0_0_2 = -1420101e91 + 0e100 = -1420101e91
// WXb_0_0 = -1420101e91 + -3243267e93 = -3257468e93
// A_1_1_0 = 5100000e94 * -2016841e93 = -1028588e94
// A_1_1_1 = 3500000e94 * 1471250e93 = 5149375e93
// A_1_1_2 = 1400000e94 * 1182671e94 = 1655739e94
// A_1_1_3 = 2000000e93 * 6379079e93 = 1275815e93
// B_0_1_0 = -1028588e94 + 5149375e93 = -5136505e93
// B_0_1_1 = -5136505e93 + 1655739e94 = 1142088e94
// B_0_1_2 = 1142088e94 + 1275815e93 = 1269669e94
// WXb_0_1 = 1269669e94 + -3389538e93 = 9307152e93
// A_2_0_0 = 0e100 * 0e100 = 0e100
// A_2_0_1 = 9307152e93 * 1024490e94 = 9535084e93
// B_1_0_0 = 0e100 + 9535084e93 = 9535084e93
// WXb_1_0 = 9535084e93 + -9810232e93 = -275148e93
// A_2_1_0 = 0e100 * 0e100 = 0e100
// A_2_1_1 = 9307152e93 * -3853514e93 = -3586524e93
// B_1_1_0 = 0e100 + -3586524e93 = -3586524e93
// WXb_1_1 = -3586524e93 + 2691283e94 = 2332630e94
// A_2_2_0 = 0e100 * 1470694e92 = 0e100
// A_2_2_1 = 9307152e93 * -4159037e93 = -3870878e93
// B_1_2_0 = 0e100 + -3870878e93 = -3870878e93
// WXb_1_2 = -3870878e93 + 1469111e94 = 1082023e94
// A_3_0_0 = 0e100 * -5066602e93 = 0e100
// A_3_0_1 = 2332630e94 * 2158575e94 = 5035156e94
// A_3_0_2 = 1082023e94 * 1194589e94 = 1292572e94
// B_2_0_0 = 0e100 + 5035156e94 = 5035156e94
// B_2_0_1 = 5035156e94 + 1292572e94 = 6327728e94
// WXb_2_0 = 6327728e94 + 6961362e92 = 6397341e94
// A_3_1_0 = 0e100 * 6979224e93 = 0e100
// A_3_1_1 = 2332630e94 * 1370535e94 = 3196951e94
// A_3_1_2 = 1082023e94 * -1185106e94 = -1282311e94
// B_2_1_0 = 0e100 + 3196951e94 = 3196951e94
// B_2_1_1 = 3196951e94 + -1282311e94 = 1914640e94
// WXb_2_1 = 1914640e94 + -2016076e93 = 1713032e94
// A_3_2_0 = 0e100 * 1254845e94 = 0e100
// A_3_2_1 = 2332630e94 * -2368272e94 = -5524302e94
// A_3_2_2 = 1082023e94 * -4329823e93 = -4684968e93
// B_2_2_0 = 0e100 + -5524302e94 = -5524302e94
// B_2_2_1 = -5524302e94 + -4684968e93 = -5992798e94
// WXb_2_2 = -5992798e94 + -9187548e93 = -6911552e94
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => 3464047e86
// A_1_0_1 => -1376897e91
// A_1_0_2 => -4323970e89
// A_1_0_3 => 0e100
// B_0_0_0 => -1376862e91
// B_0_0_1 => -1420101e91
// B_0_0_2 => -1420101e91
// WXb_0_0 => -3257468e93
// X_1_0 => 0e100
// A_1_1_0 => -1028588e94
// A_1_1_1 => 5149375e93
// A_1_1_2 => 1655739e94
// A_1_1_3 => 1275815e93
// B_0_1_0 => -5136505e93
// B_0_1_1 => 1142088e94
// B_0_1_2 => 1269669e94
// WXb_0_1 => 9307152e93
// X_1_1 => 9307152e93
// A_2_0_0 => 0e100
// A_2_0_1 => 9535084e93
// B_1_0_0 => 9535084e93
// WXb_1_0 => -275148e93
// X_2_0 => 0e100
// A_2_1_0 => 0e100
// A_2_1_1 => -3586524e93
// B_1_1_0 => -3586524e93
// WXb_1_1 => 2332630e94
// X_2_1 => 2332630e94
// A_2_2_0 => 0e100
// A_2_2_1 => -3870878e93
// B_1_2_0 => -3870878e93
// WXb_1_2 => 1082023e94
// X_2_2 => 1082023e94
// A_3_0_0 => 0e100
// A_3_0_1 => 5035156e94
// A_3_0_2 => 1292572e94
// B_2_0_0 => 5035156e94
// B_2_0_1 => 6327728e94
// WXb_2_0 => 6397341e94
// O_3_0 => 6397341e94
// A_3_1_0 => 0e100
// A_3_1_1 => 3196951e94
// A_3_1_2 => -1282311e94
// B_2_1_0 => 3196951e94
// B_2_1_1 => 1914640e94
// WXb_2_1 => 1713032e94
// O_3_1 => 1713032e94
// A_3_2_0 => 0e100
// A_3_2_1 => -5524302e94
// A_3_2_2 => -4684968e93
// B_2_2_0 => -5524302e94
// B_2_2_1 => -5992798e94
// WXb_2_2 => -6911552e94
// O_3_2 => 0e100
fn main() {
	let X_0_0 = Float{ sign: 0, mantissa: 5100000, exponent: 94 };
	let X_0_1 = Float{ sign: 0, mantissa: 3500000, exponent: 94 };
	let X_0_2 = Float{ sign: 0, mantissa: 1400000, exponent: 94 };
	let X_0_3 = Float{ sign: 0, mantissa: 2000000, exponent: 93 };
	let y_1 = Float{ sign: 0, mantissa: 6397341, exponent: 94 };
	let y_2 = Float{ sign: 0, mantissa: 1713032, exponent: 94 };
	let y_3 = Float{ sign: 0, mantissa: 0, exponent: 100 };

	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 679225, exponent: 86 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 3933992, exponent: 90 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 1, mantissa: 3088550, exponent: 89 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 0, exponent: 100 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 1, mantissa: 3243267, exponent: 93 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 1, mantissa: 2016841, exponent: 93 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 1471250, exponent: 93 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 1182671, exponent: 94 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 6379079, exponent: 93 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 1, mantissa: 3389538, exponent: 93 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 0, exponent: 100 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 1024490, exponent: 94 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 1, mantissa: 9810232, exponent: 93 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 0, exponent: 100 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 3853514, exponent: 93 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 0, mantissa: 2691283, exponent: 94 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 1470694, exponent: 92 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 4159037, exponent: 93 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 0, mantissa: 1469111, exponent: 94 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 5066602, exponent: 93 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 2158575, exponent: 94 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 1194589, exponent: 94 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 0, mantissa: 6961362, exponent: 92 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 6979224, exponent: 93 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 1370535, exponent: 94 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 1185106, exponent: 94 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 1, mantissa: 2016076, exponent: 93 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 1254845, exponent: 94 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 2368272, exponent: 94 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 4329823, exponent: 93 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 1, mantissa: 9187548, exponent: 93 });
	let O_3_2 = relu(WXb_2_2);

	assert( O_3_0.sign == y_1.sign, 'assert 1');
	assert( O_3_0.mantissa == y_1.mantissa, 'assert 2');
	assert( O_3_0.exponent == y_1.exponent, 'assert 3');
	assert( O_3_1.sign == y_2.sign, 'assert 4');
	assert( O_3_1.mantissa == y_2.mantissa, 'assert 5');
	assert( O_3_1.exponent == y_2.exponent, 'assert 6');
	assert( O_3_2.sign == y_3.sign, 'assert 7');
	assert( O_3_2.mantissa == y_3.mantissa, 'assert 8');
	assert( O_3_2.exponent == y_3.exponent, 'assert 9');
}
