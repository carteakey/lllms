use std::process::ExitCode;

fn main() -> ExitCode {
    match l3ms::cli::run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
