"""Production authentication service"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient

from backend.config import settings
from backend.schemas.models import UserCreate, APIKey
from backend.utils.logger import log

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Production authentication service"""
    
    def __init__(self, db_client: AsyncIOMotorClient):
        self.db = db_client[settings.DATABASE_NAME]
        self.users_collection = self.db.users
        self.api_keys_collection = self.db.api_keys
        
    async def create_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Create a new user"""
        
        # Check if user exists
        existing = await self.users_collection.find_one({"username": user_data.username})
        if existing:
            raise ValueError("Username already exists")
        
        existing_email = await self.users_collection.find_one({"email": user_data.email})
        if existing_email:
            raise ValueError("Email already registered")
        
        # Create user document
        user_doc = {
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role,
            "organization": user_data.organization,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user
        result = await self.users_collection.insert_one(user_doc)
        user_doc["_id"] = str(result.inserted_id)
        
        # Generate API key
        api_key = await self.generate_api_key(user_doc["_id"])
        
        log.info(f"Created user: {user_data.username}")
        
        return {
            "user": user_doc,
            "api_key": api_key
        }
    
    async def generate_api_key(self, user_id: str) -> str:
        """Generate a new API key for a user"""
        
        # Generate secure key
        key = f"sk-{secrets.token_hex(32)}"
        
        # Store in database
        api_key_doc = {
            "key": self._hash_api_key(key),
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "is_active": True
        }
        
        await self.api_keys_collection.insert_one(api_key_doc)
        
        log.info(f"Generated API key for user: {user_id}")
        
        return key
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return user info"""
        
        # Hash the key for comparison
        hashed_key = self._hash_api_key(api_key)
        
        # Find API key
        key_doc = await self.api_keys_collection.find_one({
            "key": hashed_key,
            "is_active": True
        })
        
        if not key_doc:
            return None
        
        # Update last used
        await self.api_keys_collection.update_one(
            {"_id": key_doc["_id"]},
            {"$set": {"last_used": datetime.utcnow()}}
        )
        
        # Get user
        user = await self.users_collection.find_one({
            "_id": key_doc["user_id"],
            "is_active": True
        })
        
        if not user:
            return None
        
        return {
            "user_id": str(user["_id"]),
            "username": user["username"],
            "role": user["role"],
            "organization": user.get("organization")
        }
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        
        hashed_key = self._hash_api_key(api_key)
        
        result = await self.api_keys_collection.update_one(
            {"key": hashed_key},
            {"$set": {"is_active": False, "revoked_at": datetime.utcnow()}}
        )
        
        return result.modified_count > 0
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def create_access_token(self, data: Dict) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except JWTError:
            return None