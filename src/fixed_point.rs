use std::mem;

pub fn iterate<V: PartialEq, F: Fn(&V) -> V>(initial: V, f: F) -> V {
    let mut v: V = initial;
    loop {
        let v_new = f(&v);
        let v_old = mem::replace(&mut v, v_new);
        if v == v_old {
            return v;
        }
    }
}

pub fn iterate_cmp<V: PartialEq, F: FnMut(&V) -> V, Cmp: FnMut(V, V) -> (V, bool)>(
    initial: V,
    mut cmp: Cmp,
    mut f: F,
) -> V {
    let mut v: V = initial;
    loop {
        let v_new = f(&v);
        let v_old = v;
        let (v_adjusted, stop) = cmp(v_old, v_new);
        if stop {
            return v_adjusted;
        }
        v = v_adjusted;
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed_point;

    #[test]
    fn fixed_point() {
        assert_eq!(fixed_point::iterate(10, |i| i / 2), 0);
    }
}
