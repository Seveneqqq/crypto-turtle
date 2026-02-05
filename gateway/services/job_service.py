import paramiko
from pathlib import Path
from io import StringIO
from typing import Optional, Dict, Any
import os

class JobService:

    def __init__(self, plgrid_username: str, ssh_key: Optional[str] = None):
        self.plgrid_username = plgrid_username
        self._ssh: Optional[paramiko.SSHClient] = None
        self.ssh_key = ssh_key
        self.base_dir = Path("/app").resolve()

    def _connect(self, host: str):
        if self._ssh is None:
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.ssh_key:
                pkey = paramiko.Ed25519Key.from_private_key(StringIO(self.ssh_key))
                self._ssh.connect(hostname=host, username=self.plgrid_username, pkey=pkey, timeout=10)
            else:
                self._ssh.connect(hostname=host, username=self.plgrid_username, timeout=10)

    def _run(self, host: str, command: str) -> Dict[str, Any]:
        self._connect(host)
        stdin, stdout, stderr = self._ssh.exec_command(command)
        return {"stdout": stdout.read().decode(), "stderr": stderr.read().decode()}

    def _sftp_upload_folder(self, host: str, relative_path: str, remote_path: str = "~") -> Dict[str, Any]:
        self._connect(host)
        local_path = (self.base_dir / relative_path).resolve()
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(f"Local path does not exist or is not a directory: {local_path}")

        sftp = self._ssh.open_sftp()
        try:
            try:
                sftp.chdir(remote_path)
            except IOError:
                self._ssh.exec_command(f"mkdir -p {remote_path}")
                sftp.chdir(remote_path)

            def upload_dir(local_dir: Path, remote_dir: str):
                for item in local_dir.iterdir():
                    remote_item = os.path.join(remote_dir, item.name)
                    if item.is_dir():
                        try:
                            sftp.mkdir(remote_item)
                        except IOError:
                            pass
                        upload_dir(item, remote_item)
                    else:
                        sftp.put(str(item), remote_item)

            upload_dir(local_path, remote_path)
        finally:
            sftp.close()

        return {"uploaded_path": str(local_path), "remote_path": remote_path}

    def _sftp_upload_file(self, host: str, relative_file: str, remote_path: str) -> Dict[str, Any]:
        self._connect(host)
        local_path = (self.base_dir / relative_file).resolve()
        if not local_path.exists() or not local_path.is_file():
            raise ValueError(f"Local file does not exist: {local_path}")

        sftp = self._ssh.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()
        return {"uploaded_file": str(local_path), "remote_path": remote_path}

    def handle_action(self, host: str, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        custom_command = payload.get("custom_command")

        if custom_command:
            return self._run(host, custom_command)
        if action == "healthcheck":
            return self._run(host, "hostname")
        if action == "list_files":
            path = payload.get("path", "~")
            return self._run(host, f"ls -la {path}")
        if action == "run_training":
            return self._run(host, "cd projekt && sbatch job.slurm")
        if action == "queue":
            return self._run(host, f"squeue -u {self.plgrid_username}")
        if action == "folder_upload":
            relative_path = payload.get("relative_path")
            remote_path = payload.get("remote_path", "~")
            if not relative_path:
                raise ValueError("relative_path is required for folder_upload")
            local_path = (self.base_dir / relative_path).resolve()
            if local_path.is_dir():
                return self._sftp_upload_folder(host, relative_path, remote_path)
            elif local_path.is_file():
                return self._sftp_upload_file(host, relative_path, remote_path)
            else:
                raise ValueError(f"Local path does not exist: {local_path}")

        raise ValueError(f"Unknown action: {action}")

    def close(self):
        if self._ssh:
            self._ssh.close()
            self._ssh = None
