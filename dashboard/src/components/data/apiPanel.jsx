import React from 'react';
import {
    CardContent,
    Card
} from '../ui/card';

function ApiPanel() {
  return (
    <div className="w-full max-w-xl items-start gap-4">
        <Card>
          <CardContent>
            <h2 className="text-lg font-semibold">API Panel</h2>
            <p className="text-sm text-gray-600">Manage your API settings and configurations.</p>
          </CardContent>
        </Card>
    </div>
  )
}

export default ApiPanel