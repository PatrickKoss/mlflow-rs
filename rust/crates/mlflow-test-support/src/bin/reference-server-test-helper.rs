#[cfg(target_os = "linux")]
mod linux {
    use std::io::{self, Write};
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    use mlflow_test_support::reference_server::{ProcessGroupChild, OWNER_PID_ENV};

    const MODE_ENV: &str = "MLFLOW_TEST_SUPPORT_HELPER_MODE";

    pub fn run() -> io::Result<()> {
        match std::env::var(MODE_ENV).as_deref() {
            Ok("owner") => owner(),
            Ok("intermediary") => intermediary(),
            Ok("candidate") => candidate(),
            Ok("tree") => tree(),
            Ok("leaf") => block(),
            _ => Ok(()),
        }
    }

    fn owner() -> io::Result<()> {
        let mut command = Command::new(std::env::current_exe()?);
        command
            .env(MODE_ENV, "intermediary")
            .env(OWNER_PID_ENV, std::process::id().to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::null());
        let _intermediary = ProcessGroupChild::spawn(command)?;
        block()
    }

    fn intermediary() -> io::Result<()> {
        let executable = std::env::current_exe()?;
        let mut command = Command::new(executable);
        command
            .arg0("python")
            .args([
                "-m",
                "uvicorn",
                "mlflow.server.fastapi_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "9",
                "--log-level",
                "error",
            ])
            .env(MODE_ENV, "candidate")
            .env(
                "MLFLOW_TEST_SUPPORT_HELPER_INTERMEDIARY_PID",
                std::process::id().to_string(),
            )
            .stdout(Stdio::inherit())
            .stderr(Stdio::null())
            .spawn()?;
        block()
    }

    fn candidate() -> io::Result<()> {
        let grandchild = Command::new(std::env::current_exe()?)
            .env(MODE_ENV, "leaf")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let intermediary =
            std::env::var("MLFLOW_TEST_SUPPORT_HELPER_INTERMEDIARY_PID").expect("intermediary pid");
        println!(
            "READY {} {} {intermediary}",
            std::process::id(),
            grandchild.id()
        );
        io::stdout().flush()?;
        block()
    }

    fn tree() -> io::Result<()> {
        let child = Command::new(std::env::current_exe()?)
            .env(MODE_ENV, "leaf")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        println!("READY {} {}", std::process::id(), child.id());
        io::stdout().flush()?;
        block()
    }

    fn block() -> io::Result<()> {
        loop {
            // SAFETY: pause only blocks until a signal is delivered.
            unsafe {
                libc::pause();
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {}
