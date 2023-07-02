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
// A_1_0_0 = 5100000e94 * -5422474e93 = -2765461e94
// A_1_0_1 = 3500000e94 * -1509631e93 = -5283708e93
// A_1_0_2 = 1400000e94 * 9585442e93 = 1341961e94
// A_1_0_3 = 2000000e93 * 1079883e94 = 2159766e93
// B_0_0_0 = -2765461e94 + -5283708e93 = -3293831e94
// B_0_0_1 = -3293831e94 + 1341961e94 = -1951870e94
// B_0_0_2 = -1951870e94 + 2159766e93 = -1735893e94
// WXb_0_0 = -1735893e94 + 1073421e94 = -662472e94
// A_1_1_0 = 5100000e94 * -9670264e91 = -4931834e92
// A_1_1_1 = 3500000e94 * -8315647e92 = -2910476e93
// A_1_1_2 = 1400000e94 * 7555629e86 = 1057788e87
// A_1_1_3 = 2000000e93 * 9822543e89 = 1964508e89
// B_0_1_0 = -4931834e92 + -2910476e93 = -3403659e93
// B_0_1_1 = -3403659e93 + 1057788e87 = -3403657e93
// B_0_1_2 = -3403657e93 + 1964508e89 = -3403460e93
// WXb_0_1 = -3403460e93 + -7689792e93 = -1109325e94
// A_2_0_0 = 0e100 * -7565730e93 = 0e100
// A_2_0_1 = 0e100 * -1422424e90 = 0e100
// B_1_0_0 = 0e100 + 0e100 = 0e100
// WXb_1_0 = 0e100 + 1166841e94 = 1166841e94
// A_2_1_0 = 0e100 * -173e86 = 0e100
// A_2_1_1 = 0e100 * 3326744e91 = 0e100
// B_1_1_0 = 0e100 + 0e100 = 0e100
// WXb_1_1 = 0e100 + -3018038e93 = -3018038e93
// A_2_2_0 = 0e100 * -3681168e93 = 0e100
// A_2_2_1 = 0e100 * -4147906e88 = 0e100
// B_1_2_0 = 0e100 + 0e100 = 0e100
// WXb_1_2 = 0e100 + 7345154e93 = 7345154e93
// A_3_0_0 = 1166841e94 * 2199182e94 = 2566095e94
// A_3_0_1 = 0e100 * -6320441e92 = 0e100
// A_3_0_2 = 7345154e93 * 1702118e94 = 1250231e94
// B_2_0_0 = 2566095e94 + 0e100 = 2566095e94
// B_2_0_1 = 2566095e94 + 1250231e94 = 3816326e94
// WXb_2_0 = 3816326e94 + -8068649e93 = 3009461e94
// A_3_1_0 = 1166841e94 * -6908256e93 = -8060836e93
// A_3_1_1 = 0e100 * -5093155e92 = 0e100
// A_3_1_2 = 7345154e93 * -1573242e94 = -1155570e94
// B_2_1_0 = -8060836e93 + 0e100 = -8060836e93
// B_2_1_1 = -8060836e93 + -1155570e94 = -1961653e94
// WXb_2_1 = -1961653e94 + 6239039e93 = -1337749e94
// A_3_2_0 = 1166841e94 * -1771110e94 = -2066603e94
// A_3_2_1 = 0e100 * -2374791e91 = 0e100
// A_3_2_2 = 7345154e93 * -1539892e93 = -1131074e93
// B_2_2_0 = -2066603e94 + 0e100 = -2066603e94
// B_2_2_1 = -2066603e94 + -1131074e93 = -2179710e94
// WXb_2_2 = -2179710e94 + 4816844e93 = -1698025e94
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => -2765461e94
// A_1_0_1 => -5283708e93
// A_1_0_2 => 1341961e94
// A_1_0_3 => 2159766e93
// B_0_0_0 => -3293831e94
// B_0_0_1 => -1951870e94
// B_0_0_2 => -1735893e94
// WXb_0_0 => -662472e94
// X_1_0 => 0e100
// A_1_1_0 => -4931834e92
// A_1_1_1 => -2910476e93
// A_1_1_2 => 1057788e87
// A_1_1_3 => 1964508e89
// B_0_1_0 => -3403659e93
// B_0_1_1 => -3403657e93
// B_0_1_2 => -3403460e93
// WXb_0_1 => -1109325e94
// X_1_1 => 0e100
// A_2_0_0 => 0e100
// A_2_0_1 => 0e100
// B_1_0_0 => 0e100
// WXb_1_0 => 1166841e94
// X_2_0 => 1166841e94
// A_2_1_0 => 0e100
// A_2_1_1 => 0e100
// B_1_1_0 => 0e100
// WXb_1_1 => -3018038e93
// X_2_1 => 0e100
// A_2_2_0 => 0e100
// A_2_2_1 => 0e100
// B_1_2_0 => 0e100
// WXb_1_2 => 7345154e93
// X_2_2 => 7345154e93
// A_3_0_0 => 2566095e94
// A_3_0_1 => 0e100
// A_3_0_2 => 1250231e94
// B_2_0_0 => 2566095e94
// B_2_0_1 => 3816326e94
// WXb_2_0 => 3009461e94
// O_3_0 => 3009461e94
// A_3_1_0 => -8060836e93
// A_3_1_1 => 0e100
// A_3_1_2 => -1155570e94
// B_2_1_0 => -8060836e93
// B_2_1_1 => -1961653e94
// WXb_2_1 => -1337749e94
// O_3_1 => 0e100
// A_3_2_0 => -2066603e94
// A_3_2_1 => 0e100
// A_3_2_2 => -1131074e93
// B_2_2_0 => -2066603e94
// B_2_2_1 => -2179710e94
// WXb_2_2 => -1698025e94
// O_3_2 => 0e100
fn main() {
	let X_0_0 = Float{ sign: 0, mantissa: 5100000, exponent: 94 };
	let X_0_1 = Float{ sign: 0, mantissa: 3500000, exponent: 94 };
	let X_0_2 = Float{ sign: 0, mantissa: 1400000, exponent: 94 };
	let X_0_3 = Float{ sign: 0, mantissa: 2000000, exponent: 93 };
	let y_1 = Float{ sign: 0, mantissa: 3009461, exponent: 94 };
	let y_2 = Float{ sign: 0, mantissa: 0, exponent: 100 };
	let y_3 = Float{ sign: 0, mantissa: 0, exponent: 100 };

	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 1, mantissa: 5422474, exponent: 93 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 1509631, exponent: 93 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 9585442, exponent: 93 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 1079883, exponent: 94 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 0, mantissa: 1073421, exponent: 94 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 1, mantissa: 9670264, exponent: 91 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 8315647, exponent: 92 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 7555629, exponent: 86 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 9822543, exponent: 89 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 1, mantissa: 7689792, exponent: 93 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 7565730, exponent: 93 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 1422424, exponent: 90 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 0, mantissa: 1166841, exponent: 94 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 173, exponent: 86 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 3326744, exponent: 91 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 1, mantissa: 3018038, exponent: 93 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 3681168, exponent: 93 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 4147906, exponent: 88 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 0, mantissa: 7345154, exponent: 93 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 2199182, exponent: 94 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 6320441, exponent: 92 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 1702118, exponent: 94 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 1, mantissa: 8068649, exponent: 93 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 6908256, exponent: 93 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 5093155, exponent: 92 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 1573242, exponent: 94 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 0, mantissa: 6239039, exponent: 93 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 1771110, exponent: 94 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 2374791, exponent: 91 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 1539892, exponent: 93 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 0, mantissa: 4816844, exponent: 93 });
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
