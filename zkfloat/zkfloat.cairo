// Struct representing float numbers using sign, mantissa and exponent.
// When Cairo language gets the update to support signed integers, the sign field will be removed
#[derive(Copy, Drop)]
struct Float {
    sign: u8,
    mantissa: u256,
    exponent: u256,
}

// Float number precision
const precision : u256 = 5;

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