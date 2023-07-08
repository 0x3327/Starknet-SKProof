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
// A_1_0_0 = 5100000e94 * 1118283e94 = 5703243e94
// A_1_0_1 = 3500000e94 * 4968464e93 = 1738962e94
// A_1_0_2 = 1400000e94 * -1100213e94 = -1540298e94
// A_1_0_3 = 2000000e93 * -1175861e94 = -2351722e93
// B_0_0_0 = 5703243e94 + 1738962e94 = 7442205e94
// B_0_0_1 = 7442205e94 + -1540298e94 = 5901907e94
// B_0_0_2 = 5901907e94 + -2351722e93 = 5666734e94
// WXb_0_0 = 5666734e94 + 8198805e93 = 6486614e94
// A_1_1_0 = 5100000e94 * 1170625e94 = 5970187e94
// A_1_1_1 = 3500000e94 * -1339560e94 = -4688460e94
// A_1_1_2 = 1400000e94 * 4150803e93 = 5811124e93
// A_1_1_3 = 2000000e93 * 1486243e94 = 2972486e93
// B_0_1_0 = 5970187e94 + -4688460e94 = 1281727e94
// B_0_1_1 = 1281727e94 + 5811124e93 = 1862839e94
// B_0_1_2 = 1862839e94 + 2972486e93 = 2160087e94
// WXb_0_1 = 2160087e94 + -4544296e93 = 1705657e94
// A_2_0_0 = 6486614e94 * -1463695e94 = -9494424e94
// A_2_0_1 = 1705657e94 * 5688212e93 = 9702138e93
// B_1_0_0 = -9494424e94 + 9702138e93 = -8524210e94
// WXb_1_0 = -8524210e94 + 3104095e93 = -8213800e94
// A_2_1_0 = 6486614e94 * -4517394e93 = -2930259e94
// A_2_1_1 = 1705657e94 * 6733924e93 = 1148576e94
// B_1_1_0 = -2930259e94 + 1148576e94 = -1781683e94
// WXb_1_1 = -1781683e94 + 9287307e93 = -8529523e93
// A_2_2_0 = 6486614e94 * 2441384e91 = 1583631e92
// A_2_2_1 = 1705657e94 * -2200086e92 = -3752592e92
// B_1_2_0 = 1583631e92 + -3752592e92 = -2168961e92
// WXb_1_2 = -2168961e92 + -6575089e93 = -6791985e93
// A_3_0_0 = 0e100 * -7529542e93 = 0e100
// A_3_0_1 = 0e100 * -1213658e94 = 0e100
// A_3_0_2 = 0e100 * -3200535e89 = 0e100
// B_2_0_0 = 0e100 + 0e100 = 0e100
// B_2_0_1 = 0e100 + 0e100 = 0e100
// WXb_2_0 = 0e100 + 1517988e94 = 1517988e94
// A_3_1_0 = 0e100 * -1221788e94 = 0e100
// A_3_1_1 = 0e100 * 8858901e93 = 0e100
// A_3_1_2 = 0e100 * 2207043e92 = 0e100
// B_2_1_0 = 0e100 + 0e100 = 0e100
// B_2_1_1 = 0e100 + 0e100 = 0e100
// WXb_2_1 = 0e100 + -8762696e93 = -8762696e93
// A_3_2_0 = 0e100 * 1800546e94 = 0e100
// A_3_2_1 = 0e100 * -5951817e92 = 0e100
// A_3_2_2 = 0e100 * 6244912e92 = 0e100
// B_2_2_0 = 0e100 + 0e100 = 0e100
// B_2_2_1 = 0e100 + 0e100 = 0e100
// WXb_2_2 = 0e100 + -1559046e94 = -1559046e94
// X_0_0 => 5100000e94
// X_0_1 => 3500000e94
// X_0_2 => 1400000e94
// X_0_3 => 2000000e93
// A_1_0_0 => 5703243e94
// A_1_0_1 => 1738962e94
// A_1_0_2 => -1540298e94
// A_1_0_3 => -2351722e93
// B_0_0_0 => 7442205e94
// B_0_0_1 => 5901907e94
// B_0_0_2 => 5666734e94
// WXb_0_0 => 6486614e94
// X_1_0 => 6486614e94
// A_1_1_0 => 5970187e94
// A_1_1_1 => -4688460e94
// A_1_1_2 => 5811124e93
// A_1_1_3 => 2972486e93
// B_0_1_0 => 1281727e94
// B_0_1_1 => 1862839e94
// B_0_1_2 => 2160087e94
// WXb_0_1 => 1705657e94
// X_1_1 => 1705657e94
// A_2_0_0 => -9494424e94
// A_2_0_1 => 9702138e93
// B_1_0_0 => -8524210e94
// WXb_1_0 => -8213800e94
// X_2_0 => 0e100
// A_2_1_0 => -2930259e94
// A_2_1_1 => 1148576e94
// B_1_1_0 => -1781683e94
// WXb_1_1 => -8529523e93
// X_2_1 => 0e100
// A_2_2_0 => 1583631e92
// A_2_2_1 => -3752592e92
// B_1_2_0 => -2168961e92
// WXb_1_2 => -6791985e93
// X_2_2 => 0e100
// A_3_0_0 => 0e100
// A_3_0_1 => 0e100
// A_3_0_2 => 0e100
// B_2_0_0 => 0e100
// B_2_0_1 => 0e100
// WXb_2_0 => 1517988e94
// O_3_0 => 1517988e94
// A_3_1_0 => 0e100
// A_3_1_1 => 0e100
// A_3_1_2 => 0e100
// B_2_1_0 => 0e100
// B_2_1_1 => 0e100
// WXb_2_1 => -8762696e93
// O_3_1 => 0e100
// A_3_2_0 => 0e100
// A_3_2_1 => 0e100
// A_3_2_2 => 0e100
// B_2_2_0 => 0e100
// B_2_2_1 => 0e100
// WXb_2_2 => -1559046e94
// O_3_2 => 0e100
fn main() {
	let X_0_0 = Float{ sign: 0, mantissa: 5100000, exponent: 94 };
	let X_0_1 = Float{ sign: 0, mantissa: 3500000, exponent: 94 };
	let X_0_2 = Float{ sign: 0, mantissa: 1400000, exponent: 94 };
	let X_0_3 = Float{ sign: 0, mantissa: 2000000, exponent: 93 };
	let y_1 = Float{ sign: 0, mantissa: 1517988, exponent: 94 };
	let y_2 = Float{ sign: 0, mantissa: 0, exponent: 100 };
	let y_3 = Float{ sign: 0, mantissa: 0, exponent: 100 };

	let A_1_0_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 1118283, exponent: 94 });
	let A_1_0_1 = mulFloats(X_0_1, Float {sign: 0, mantissa: 4968464, exponent: 93 });
	let A_1_0_2 = mulFloats(X_0_2, Float {sign: 1, mantissa: 1100213, exponent: 94 });
	let A_1_0_3 = mulFloats(X_0_3, Float {sign: 1, mantissa: 1175861, exponent: 94 });
	let B_0_0_0 = addFloats(A_1_0_0, A_1_0_1);
	let B_0_0_1 = addFloats(B_0_0_0, A_1_0_2);
	let B_0_0_2 = addFloats(B_0_0_1, A_1_0_3);
	let WXb_0_0 = addFloats(B_0_0_2, Float {sign: 0, mantissa: 8198805, exponent: 93 });
	let X_1_0 = relu(WXb_0_0);
	let A_1_1_0 = mulFloats(X_0_0, Float {sign: 0, mantissa: 1170625, exponent: 94 });
	let A_1_1_1 = mulFloats(X_0_1, Float {sign: 1, mantissa: 1339560, exponent: 94 });
	let A_1_1_2 = mulFloats(X_0_2, Float {sign: 0, mantissa: 4150803, exponent: 93 });
	let A_1_1_3 = mulFloats(X_0_3, Float {sign: 0, mantissa: 1486243, exponent: 94 });
	let B_0_1_0 = addFloats(A_1_1_0, A_1_1_1);
	let B_0_1_1 = addFloats(B_0_1_0, A_1_1_2);
	let B_0_1_2 = addFloats(B_0_1_1, A_1_1_3);
	let WXb_0_1 = addFloats(B_0_1_2, Float {sign: 1, mantissa: 4544296, exponent: 93 });
	let X_1_1 = relu(WXb_0_1);
	let A_2_0_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 1463695, exponent: 94 });
	let A_2_0_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 5688212, exponent: 93 });
	let B_1_0_0 = addFloats(A_2_0_0, A_2_0_1);
	let WXb_1_0 = addFloats(B_1_0_0, Float {sign: 0, mantissa: 3104095, exponent: 93 });
	let X_2_0 = relu(WXb_1_0);
	let A_2_1_0 = mulFloats(X_1_0, Float {sign: 1, mantissa: 4517394, exponent: 93 });
	let A_2_1_1 = mulFloats(X_1_1, Float {sign: 0, mantissa: 6733924, exponent: 93 });
	let B_1_1_0 = addFloats(A_2_1_0, A_2_1_1);
	let WXb_1_1 = addFloats(B_1_1_0, Float {sign: 0, mantissa: 9287307, exponent: 93 });
	let X_2_1 = relu(WXb_1_1);
	let A_2_2_0 = mulFloats(X_1_0, Float {sign: 0, mantissa: 2441384, exponent: 91 });
	let A_2_2_1 = mulFloats(X_1_1, Float {sign: 1, mantissa: 2200086, exponent: 92 });
	let B_1_2_0 = addFloats(A_2_2_0, A_2_2_1);
	let WXb_1_2 = addFloats(B_1_2_0, Float {sign: 1, mantissa: 6575089, exponent: 93 });
	let X_2_2 = relu(WXb_1_2);
	let A_3_0_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 7529542, exponent: 93 });
	let A_3_0_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 1213658, exponent: 94 });
	let A_3_0_2 = mulFloats(X_2_2, Float {sign: 1, mantissa: 3200535, exponent: 89 });
	let B_2_0_0 = addFloats(A_3_0_0, A_3_0_1);
	let B_2_0_1 = addFloats(B_2_0_0, A_3_0_2);
	let WXb_2_0 = addFloats(B_2_0_1, Float {sign: 0, mantissa: 1517988, exponent: 94 });
	let O_3_0 = relu(WXb_2_0);
	let A_3_1_0 = mulFloats(X_2_0, Float {sign: 1, mantissa: 1221788, exponent: 94 });
	let A_3_1_1 = mulFloats(X_2_1, Float {sign: 0, mantissa: 8858901, exponent: 93 });
	let A_3_1_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 2207043, exponent: 92 });
	let B_2_1_0 = addFloats(A_3_1_0, A_3_1_1);
	let B_2_1_1 = addFloats(B_2_1_0, A_3_1_2);
	let WXb_2_1 = addFloats(B_2_1_1, Float {sign: 1, mantissa: 8762696, exponent: 93 });
	let O_3_1 = relu(WXb_2_1);
	let A_3_2_0 = mulFloats(X_2_0, Float {sign: 0, mantissa: 1800546, exponent: 94 });
	let A_3_2_1 = mulFloats(X_2_1, Float {sign: 1, mantissa: 5951817, exponent: 92 });
	let A_3_2_2 = mulFloats(X_2_2, Float {sign: 0, mantissa: 6244912, exponent: 92 });
	let B_2_2_0 = addFloats(A_3_2_0, A_3_2_1);
	let B_2_2_1 = addFloats(B_2_2_0, A_3_2_2);
	let WXb_2_2 = addFloats(B_2_2_1, Float {sign: 1, mantissa: 1559046, exponent: 94 });
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
