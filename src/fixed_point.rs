use std::mem;

pub fn fixed_point_iterate<V: PartialEq, F: Fn(&V) -> V>(initial: V, f: F) -> V {
    let mut v: V = initial;
    loop {
        let v_new = f(&v);
        let v_old = mem::replace(&mut v, v_new);
        if v == v_old {
            return v;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed_point::{fixed_point_iterate};

    #[test]
    fn fixed_point() {
        assert_eq!(fixed_point_iterate(10, |i| i / 2), 0);
    }
}
