from typing import List, Dict, Any, BinaryIO
from loguru import logger
import asyncio
from datetime import datetime
import uuid
import os
from minio import Minio
from minio.error import S3Error
from io import BytesIO

from config import settings


class MinIOClient:
    """MinIO client for object storage operations."""
    
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        self.bucket_name = settings.minio_bucket_name
        
    async def initialize(self):
        """Initialize MinIO client and ensure bucket exists."""
        try:
            # Run in executor since minio client is synchronous
            loop = asyncio.get_event_loop()
            
            # Check if bucket exists
            bucket_exists = await loop.run_in_executor(
                None, self.client.bucket_exists, self.bucket_name
            )
            
            if not bucket_exists:
                # Create bucket
                await loop.run_in_executor(
                    None, self.client.make_bucket, self.bucket_name
                )
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if MinIO is healthy."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.client.bucket_exists, self.bucket_name
            )
            return True
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return False
    
    async def upload_file(self, file_path: str, object_name: str = None) -> str:
        """
        Upload a file to MinIO.
        
        Args:
            file_path: Path to the file to upload
            object_name: Object name in MinIO (defaults to filename with timestamp)
            
        Returns:
            Object name/ID in MinIO
        """
        try:
            # Generate object name if not provided
            if not object_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = os.path.basename(file_path)
                object_name = f"{timestamp}_{filename}"
            
            # Get file stats
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            
            # Upload file
            loop = asyncio.get_event_loop()
            
            with open(file_path, 'rb') as file_data:
                await loop.run_in_executor(
                    None,
                    self.client.put_object,
                    self.bucket_name,
                    object_name,
                    file_data,
                    file_size
                )
            
            logger.info(f"Uploaded file: {object_name} ({file_size} bytes)")
            return object_name
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    async def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload bytes data to MinIO.
        """
        try:
            loop = asyncio.get_event_loop()
            
            data_stream = BytesIO(data)
            await loop.run_in_executor(
                None,
                self.client.put_object,
                self.bucket_name,
                object_name,
                data_stream,
                len(data),
                content_type=content_type
            )
            
            logger.info(f"Uploaded bytes: {object_name} ({len(data)} bytes)")
            return object_name
            
        except Exception as e:
            logger.error(f"Failed to upload bytes: {e}")
            raise
    
    async def download_file(self, object_name: str, file_path: str):
        """
        Download a file from MinIO.
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            await loop.run_in_executor(
                None,
                self.client.fget_object,
                self.bucket_name,
                object_name,
                file_path
            )
            
            logger.info(f"Downloaded file: {object_name} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise
    
    async def get_file_bytes(self, object_name: str) -> bytes:
        """
        Get file contents as bytes.
        """
        try:
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                self.client.get_object,
                self.bucket_name,
                object_name
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get file bytes: {e}")
            raise
    
    async def list_files(self, prefix: str = "", recursive: bool = True) -> List[Dict[str, Any]]:
        """
        List files in the bucket.
        """
        try:
            loop = asyncio.get_event_loop()
            
            objects = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_objects(
                    self.bucket_name,
                    prefix=prefix,
                    recursive=recursive
                ))
            )
            
            files = []
            for obj in objects:
                files.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag
                })
            
            logger.info(f"Listed {len(files)} files in bucket")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
    
    async def delete_file(self, object_name: str):
        """
        Delete a file from MinIO.
        """
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                self.client.remove_object,
                self.bucket_name,
                object_name
            )
            
            logger.info(f"Deleted file: {object_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise
    
    async def file_exists(self, object_name: str) -> bool:
        """
        Check if a file exists in MinIO.
        """
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                self.client.stat_object,
                self.bucket_name,
                object_name
            )
            return True
            
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            raise
    
    async def get_file_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get file metadata.
        """
        try:
            loop = asyncio.get_event_loop()
            
            stat = await loop.run_in_executor(
                None,
                self.client.stat_object,
                self.bucket_name,
                object_name
            )
            
            return {
                "name": object_name,
                "size": stat.size,
                "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": dict(stat.metadata) if stat.metadata else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise