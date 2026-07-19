#[cfg(unix)]
fn main() -> std::io::Result<()> {
    let report = mlflow_test_support::reference_server::reap_stale_reference_servers()?;
    println!(
        "matched={} reaped={} groups={} legacy={}",
        report.matched, report.reaped, report.groups, report.legacy
    );
    Ok(())
}

#[cfg(not(unix))]
fn main() {
    eprintln!("reference-server reaping is only available on Unix");
}
