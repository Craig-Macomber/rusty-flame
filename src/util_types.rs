//! Misc wrapper types for providing Clone, Debug and Eq

use std::{ops::Deref, rc::Rc};

#[derive(Debug, Clone)]
/// Wrapper that provides Debug.
pub struct DebugIt<T>(pub T);

#[derive(Debug)]
/// Rc with pointer semantics (reference equality)
pub struct PtrRc<T>(Rc<T>);

// TODO: why does derive clone not work for this?
impl<T> Clone for PtrRc<T> {
    fn clone(&self) -> Self {
        PtrRc(self.0.clone())
    }
}

impl<T> PartialEq for PtrRc<T> {
    fn ne(&self, other: &Self) -> bool {
        !Rc::ptr_eq(&self.0, &other.0)
    }

    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Deref for PtrRc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Deref for DebugIt<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Eq for PtrRc<T> {}

impl<T> From<T> for PtrRc<T> {
    fn from(t: T) -> Self {
        PtrRc(Rc::new(t))
    }
}
