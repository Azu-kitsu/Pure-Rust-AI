use std::{fmt};
use serde::{Serialize, Deserialize};
//learned from https://www.mathsisfun.com/algebra/matrix-multiplying.html and struct solution inspired by Mark Kraay :)

#[derive(Serialize, Deserialize, Debug)]
pub struct Matrix {
    pub entries: Vec::<f64>,
    pub row: usize,
    pub col: usize,
}

impl Matrix {
    pub fn new(entries: Vec::<f64>, row: usize, col: usize) -> Matrix {
        Matrix { entries, row, col }
    }
    pub fn new_empty(row: usize, col: usize) -> Matrix {
        Matrix { entries: vec![0.0; row*col], row, col }
    }

    pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
        if a.row == b.row && a.col == b.col {
            Matrix { entries: a.entries.iter().zip(b.entries.iter()).map(|(x, y)| *x + *y).collect(), row: a.row, col: a.col}
        }
        else {
            panic!("the two matrices either didn't have matching number of rows or columns");
        }
    }

    pub fn sub(a: &Matrix, b: &Matrix) -> Matrix {
        if a.row == b.row && a.col == b.col {
            Matrix { entries: a.entries.iter().zip(b.entries.iter()).map(|(x, y)| *x - *y).collect(), row: a.row, col: a.col}
        }
        else {
            panic!("the two matrices either didn't have matching number of rows or columns");
        }
    }

    pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
        let mut new = a.iter().zip(b.iter()).map(|(x, y)| *x * *y);
        let mut elem = new.next().unwrap();
        for e in new {
            elem = elem + e;
        }
        elem    
    }

    pub fn hadamard_product(a: &Matrix, b: &Matrix) -> Matrix {
        let mut entries = Vec::<f64>::new();
        for (x, y) in a.entries.iter().zip(b.entries.iter()) {
            entries.push(x * y);
        }
        Matrix { entries, row: a.row, col: a.col }
    }
    
    pub fn get_rows(&self) -> Vec<Vec<f64>> {
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(self.row);
        let chunks = self.entries.chunks(self.col as usize);
        for row in chunks {
            rows.push(row.to_vec());
        }
        rows
    }

    pub fn get_cols(&self) -> Vec<Vec<f64>> {
        let mut cols: Vec<Vec<f64>> = Vec::with_capacity(self.col);
        for _ in 0..self.col {
            cols.push(Vec::new());
        }
        for (id, elem) in self.entries.iter().enumerate() {
            let ord = id % self.col;
            cols[ord].push(elem.clone());
        }
        cols
    }

    pub fn transpose(&self) -> Matrix {
        let cols = self.get_cols();
        let mut entries: Vec<f64> = Vec::new();
        for col in cols.iter() {entries.extend(col);}
        Matrix { entries: entries, row: self.col, col: self.row}
    }

    pub fn multiply(&self, second: &Matrix) -> Matrix { //column of weights != main rows
        if self.col == second.row {
            let mut result = Matrix {entries: Vec::<f64>::new(), row: self.row, col: second.col};
            let rows_self = self.get_rows();
            let cols_sec = second.get_cols();
            for row in rows_self.iter() {
                for col in cols_sec.iter() {
                    let prod = Matrix::dot_product(row.clone(), col.clone());
                    result.entries.push(prod);
                }
            }
            result
        }
        else {
            panic!("the first vector's column number does not match the second one's row number");
        }
    }

    pub fn flatten(&self) -> Matrix {
        Matrix {entries: self.entries.clone(), row: 1, col: self.row * self.col}
    }
    
    pub fn pancake(&self) -> Matrix {
        Matrix {entries: self.entries.clone(), row: self.row * self.col, col: 1}
    }
    
    pub fn add_scalar(&self, scalar: f64) -> Matrix {
        Matrix {entries: self.entries.iter().map(|x| *x + scalar).collect(), row: self.row, col: self.col}
    }
    
    pub fn scale(&self, scalar: f64) -> Matrix {
        Matrix {entries: self.entries.iter().map(|x| *x * scalar).collect(), row: self.row, col: self.col}
    }

    pub fn apply(&mut self, f: &dyn Fn(f64) -> f64) {
            self.entries = self.entries
            .iter()
            .map(|x| f(*x))
            .collect();
    }

    pub fn apply_new(&self, f: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix { 
            entries: self.entries.iter()
            .map(|x| f(*x))
            .collect(),
            row: self.row, col: self.col
         }
    }

    pub fn apply_new_2(&self, p: f64, f: &dyn Fn(f64, f64) -> f64) -> Matrix {
        Matrix { 
            entries: self.entries.iter()
            .map(|x| f(*x, p))
            .collect(),
            row: self.row, col: self.col
         }
    }

    pub fn resize(&mut self, row: usize, col: usize) {
        if row*col == self.row*self.col {
            self.row = row;
            self.col = col;
        }
        else {
            panic!("during a resize operation the product of the new rows and columns didn't match the original");
        }
    }

    pub fn print_image(&self) {
        let mut c = 0;
        print!("\x1B[s");
        print!("----{{matrix begin}}----");
        for row in 0..self.row {
            print!("\n\x1B[u\x1B[{}B", row+1);
            for _col in 0..self.col {
                print!("\x1B[38;2;{};{0};{0}m█", self.entries[c] * 255.0);
                c += 1;
            }
        }
        print!("\x1B[38;2;255;255;255m\n\x1B[u\x1B[{}B----{{matrix end}}---- rows: {}, columns: {}\n", self.row + 1, self.row, self.col);
    }

}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        let new_vec = self.entries.clone();
        Matrix::new(new_vec, self.row, self.col)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut c = 0;
        write!(f, "\x1B[s");
        write!(f, "----{{matrix begin}}----");
        for row in 0..self.row {
            write!(f, "\n\x1B[u\x1B[{}B", row+1);
            for _col in 0..self.col {
                write!(f, "{} ", self.entries[c]);
                c += 1;
            }
        }
        write!(f, "\n\x1B[u\x1B[{}B----{{matrix end}}---- rows: {}, columns: {}\n", self.row + 1, self.row, self.col);
        Ok(())
    }
}

//impl fmt::Display for Matrix {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        let mut c = 0;
//        write!(f, "\x1B[s");
//        write!(f, "----{{matrix begin}}----");
//        for row in 0..self.row {
//            write!(f, "\n\x1B[u\x1B[{}B", row+1);
//            for _col in 0..self.col {
//                write!(f, "\x1B[38;2;{};{0};{0}m█", self.entries[c] * 255.0, );
//                c += 1;
//            }
//        }
//        write!(f, "\n\x1B[u\x1B[{}B----{{matrix end}}---- rows: {}, columns: {}\n", self.row + 1, self.row, self.col);
//        Ok(())
//    }
//}