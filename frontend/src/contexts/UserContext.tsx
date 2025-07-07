import React, { createContext, useState, useEffect, ReactNode, useContext } from 'react';
import { User, UserProfile } from '@/types/user';
import { AuthContext } from './AuthContext';

interface UserContextType {
  profile: UserProfile | null;
  isLoading: boolean;
  error: string | null;
  updateProfile: (data: Partial<UserProfile>) => Promise<void>;
  updateSettings: (settings: UserProfile['settings']) => Promise<void>;
}

const defaultValue: UserContextType = {
  profile: null,
  isLoading: false,
  error: null,
  updateProfile: async () => {},
  updateSettings: async () => {},
};

export const UserContext = createContext<UserContextType>(defaultValue);

interface UserProviderProps {
  children: ReactNode;
}

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const { user, isAuthenticated } = useContext(AuthContext);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch user profile when authenticated
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!isAuthenticated || !user) {
        setProfile(null);
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        // In a real app, we would fetch the profile from the backend
        await new Promise(resolve => setTimeout(resolve, 500));

        // Create a mock profile based on the authenticated user
        const mockProfile: UserProfile = {
          ...user,
          bio: '数字分身平台用户，热爱技术与创新。',
          phone: '+86 123 4567 8901',
          skills: ['React', 'TypeScript', 'Node.js', '数据分析', '项目管理'],
          social: {
            linkedin: 'https://linkedin.com/in/username',
            github: 'https://github.com/username',
            twitter: 'https://twitter.com/username',
            website: 'https://example.com',
          },
          settings: {
            theme: 'light',
            language: 'zh-CN',
            notifications: {
              email: true,
              push: true,
              sms: false,
            },
            privacy: {
              showProfile: true,
              showEmail: false,
              showActivity: true,
            },
          },
        };

        setProfile(mockProfile);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserProfile();
  }, [user, isAuthenticated]);

  const updateProfile = async (data: Partial<UserProfile>) => {
    if (!profile) return;

    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the updated profile to the backend
      await new Promise(resolve => setTimeout(resolve, 800));

      // Update the profile locally
      const updatedProfile = {
        ...profile,
        ...data,
        updatedAt: new Date(),
      };

      setProfile(updatedProfile);

      // Update local storage to reflect changes
      if (user) {
        const updatedUser = {
          ...user,
          firstName: updatedProfile.firstName,
          lastName: updatedProfile.lastName,
          displayName: updatedProfile.displayName,
          avatar: updatedProfile.avatar,
          updatedAt: updatedProfile.updatedAt,
        };
        localStorage.setItem('user', JSON.stringify(updatedUser));
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const updateSettings = async (settings: UserProfile['settings']) => {
    if (!profile || !settings) return;

    setIsLoading(true);
    setError(null);

    try {
      // In a real app, we would send the updated settings to the backend
      await new Promise(resolve => setTimeout(resolve, 500));

      // Update the profile locally
      const updatedProfile = {
        ...profile,
        settings: {
          ...profile.settings,
          ...settings,
        },
        updatedAt: new Date(),
      };

      setProfile(updatedProfile);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <UserContext.Provider
      value={{
        profile,
        isLoading,
        error,
        updateProfile,
        updateSettings,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}; 