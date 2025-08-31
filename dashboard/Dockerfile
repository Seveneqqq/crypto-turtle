FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5173

RUN sed -i 's/80;/5173;/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]